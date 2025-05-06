"""
mm_gnlse.py

A production-level package for simulating the 3D generalized nonlinear Schrödinger equation (NLSE)
in multimode fibers. The package is organized into three main classes:

  - Fiber: Stores fiber parameters and provides methods for creating transverse refractive index
           profiles (step-index or GRIN) along with (stub) mode-solver functions.
  - InputSource: Generates initial field profiles (pulsed Gaussian, CW Gaussian, or modal superposition).
  - GNLSE_Sim: Combines a Fiber and an InputSource together with simulation grid parameters,
              sets up Fourier-domain quantities, implements the NLSE evolution using diffrax,
              and provides visualization utilities.

Also included are two global helper functions used to compute the complex NLSE right‐hand side,
and to convert between a complex 3D field and a flat real vector.
"""

import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation

import scipy.sparse as sp
import scipy.sparse.linalg as spla


from scipy.ndimage import uniform_filter1d

###############################################################################
# Global Helper Functions
###############################################################################
def effective_radius_vs_z(sim, t_index=0):
    """Return w(z) for a GNLSE_Sim with Nt==1 or CW data."""
    dx = float(sim.x[1] - sim.x[0])
    dy = float(sim.y[1] - sim.y[0])

    # (Nx,Ny) radius‑squared grid – always in x,y order
    X, Y = np.meshgrid(sim.x, sim.y, indexing="ij")
    r2   = X**2 + Y**2

    w_vec = []
    for st in sim.state_samples:
        A = sim.reconstruct_field(st, sim.Nx, sim.Ny, sim.Nt)
        I = np.abs(A[:, :, t_index])**2        # intensity (Nx,Ny)
        P = np.sum(I) * dx * dy               # total power
        w2 = np.sum(r2 * I) * dx * dy / P     # <r²>
        w_vec.append(np.sqrt(w2))
    return np.asarray(w_vec)


def self_focusing_length(z_vec,
                         metric,
                         method="first_peak",
                         smooth=None,
                         peak_prominence=0.05,
                         threshold_factor=2.0):
    """
    Determine a single self‑focusing length L_sf for one simulation.

    Parameters
    ----------
    z_vec : (N,) array
        Saved propagation positions.
    metric : (N,) array
        Diagnostic versus z:
          * for "first_peak"  -> pass I_peak(z);
          * for "radius_min" -> pass beam radius w(z);
          * for "threshold"  -> pass I_peak(z) as well.
    method : {"first_peak", "radius_min", "threshold"}
    smooth : int or None
        Optional boxcar window (in samples) for metric smoothing.
    peak_prominence : float
        Minimum relative drop after the first peak (used by "first_peak").
    threshold_factor : float
        I_peak must exceed  threshold_factor × I_peak_lin(z)  ("threshold").

    Returns
    -------
    L_sf : float or None
        Self‑focusing length in metres, or None if no onset detected.
    """

    m = np.asarray(metric)

    if smooth is not None and smooth > 1:
        m = uniform_filter1d(m, size=smooth)

    # ----------  1) first pronounced peak in I_peak(z)  ----------
    if method == "first_peak":
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(m, prominence=peak_prominence * m.max())
        return z_vec[peaks[0]] if len(peaks) else None

    # ----------  2) first minimum of beam radius w(z) ------------
    elif method == "radius_min":
        idx = np.argmin(m)
        return z_vec[idx]

    # ----------  3) first crossing of a fixed threshold ----------
    elif method == "threshold":
        above = m > threshold_factor * m[0]          # reference can be any linear run
        return z_vec[np.argmax(above)] if above.any() else None

    else:
        raise ValueError("Unknown method.")


def build_1d_laplacian(N: int, dx: float) -> sp.csr_matrix:
    """
    Second‑order finite‑difference Laplacian with Dirichlet boundaries.

    Returns an N×N sparse CSR matrix.

    Δf ≈ (f_{i+1} - 2 f_i + f_{i-1}) / dx²
    """
    main = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N - 1)
    # diags automatically pads the shorter off‑diagonals
    return sp.diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N), format="csr") / dx**2


# ---------- in mm_gnlse.py (top helper section) ----------------------
def radial_index_profile(r,
                         r_core,
                         r_clad,
                         n_core,
                         n_clad,
                         n_air=1.0,
                         transition="step",   # <- NEW
                         sg_order=6):         #    super‑Gaussian order m
    """
    Core/cladding/air profile with optional smooth transition.

    transition:
        "step"           – old behaviour
        "supergaussian"  – n(r) = n_clad + (n_core-n_clad)*exp(-(r/r_c)^(2m))
    """
    if transition == "step":
        n = np.where(r <= r_core, n_core,
             np.where(r <= r_clad, n_clad, n_air))

    elif transition == "supergaussian":
        n_cg = n_clad + (n_core - n_clad) * np.exp(-(r / r_core)**(2*sg_order))
        n = np.where(r <= r_clad, n_cg, n_air)

    else:
        raise ValueError("transition must be 'step' or 'supergaussian'")

    return n



def compute_complex_rhs(A, args):
    """
    Compute the complex derivative dA/dz for the field A(x,y,t) according to the NLSE:
    
       dA/dz = i*(term_diffraction + term_dispersion - term_potential + term_nonlinear)
               - W_full*A,
    
    where:
      - term_diffraction uses the transverse Laplacian (via FFTs),
      - term_dispersion uses the second derivative in time,
      - term_potential = V(x,y)*A,
      - term_nonlinear = gamma*|A|^2*A,
      - W_full is a spatially varying damping (PML).
      
    Parameters:
       A: complex field, shape (Nx, Ny, Nt).
       args: dictionary with keys: "beta0", "beta2", "lap_op", "omega", "V",
             "gamma", "W_full".
             
    Returns:
       dA_dz: complex array of the same shape as A.
    """
    beta0 = args["beta0"]
    beta2 = args["beta2"]
    lap_op = args["lap_op"]
    omega = args["omega"]
    V = args["V"]
    gamma_space = args["gamma"]
    W_full = args["W_full"]
    
    # Diffraction term (FFT over x and y)
    A_xy_fft = jnp.fft.fftn(A, axes=(0, 1))
    A_xy_fft = jax.lax.stop_gradient(A_xy_fft)
    lap_term = A_xy_fft * lap_op[:, :, None]
    lap_term = jax.lax.stop_gradient(lap_term)
    term_diffraction = (1/(2 * beta0)) * jnp.fft.ifftn(lap_term, axes=(0, 1))
    
    # Dispersion term (FFT over t)
    A_t_fft = jnp.fft.fft(A, axis=2)
    A_t_fft = jax.lax.stop_gradient(A_t_fft)
    disp_multiplier = - (omega**2)[None, None, :]
    term_dispersion = - (beta2/2) * jnp.fft.ifft(A_t_fft * disp_multiplier, axis=2)
    
    # Potential and nonlinear terms.
    term_potential = V[:, :, None] * A
    nonlinear_phase = gamma_space[:, :, None] * jnp.abs(A)**2
    term_nonlinear = nonlinear_phase * A
    
    # Combine contributions.
    dA_dz = 1j * (term_diffraction + term_dispersion - term_potential + term_nonlinear) - W_full * A
    return dA_dz

def gnlse_rhs_cartesian(z, y_flat, args):
    """
    ODE right-hand side for the NLSE in Cartesian coordinates.
    
    The complex field A(x,y,t) is stored as a flat real vector:
      state = [vec(Re{A}), vec(Im{A})].
    
    This function reconstructs A, computes dA/dz via compute_complex_rhs,
    then splits dA/dz into its real and imaginary parts and concatenates them.
    
    Parameters:
      z: Propagation coordinate (m). (Not explicitly used if the system is z-autonomous.)
      y_flat: Flat real state vector of length 2*N_total, with N_total = Nx*Ny*Nt.
      args: Dictionary of additional parameters.
      
    Returns:
      A flat real vector representing dA/dz.
    """
    Nx = args["Nx"]
    Ny = args["Ny"]
    Nt = args["Nt"]
    N_total = Nx * Ny * Nt
    A_re = y_flat[:N_total].reshape((Nx, Ny, Nt))
    A_im = y_flat[N_total:].reshape((Nx, Ny, Nt))
    A = A_re + 1j * A_im
    dA_dz = compute_complex_rhs(A, args)
    dA_re = jnp.real(dA_dz)
    dA_im = jnp.imag(dA_dz)
    return jnp.concatenate([dA_re.ravel(), dA_im.ravel()])

###############################################################################
# 1. Fiber Class
###############################################################################
class Fiber:
    def __init__(self, core_radius, n_core, n_clad, beta0, beta2, gamma, length,
                 fiber_type='step-index', Lx=None, Ly=None, Nx=None, Ny=None, 
                 k0=None, transition="step",     
                 sg_order=6): 
        """
        Fiber object encapsulating fiber parameters.
        
        Parameters:
          core_radius: Core radius (m).
          n_core: Core refractive index.
          n_clad: Cladding refractive index.
          beta0: Propagation constant (1/m).
          beta2: Group velocity dispersion (s²/m).
          gamma: Nonlinear coefficient (W⁻¹·m⁻¹).
          length: Fiber length (m).
          fiber_type: 'step-index' or 'GRIN'.
          Lx, Ly: Transverse simulation window dimensions (m).
          Nx, Ny: Number of grid points in x and y.
          k0: Central wavenumber (1/m); if None, set to 2π/1.064e-6.
        """
        self.core_radius = core_radius
        self.n_core = n_core
        self.n_clad = n_clad
        self.beta0 = beta0
        self.beta2 = beta2
        self.gamma = gamma
        self.length = length
        self.fiber_type = fiber_type
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        if k0 is None:
            self.k0 = 2 * jnp.pi / 1.064e-6
        else:
            self.k0 = k0
        self.transition = transition
        self.sg_order   = sg_order
        
    def refractive_profile(self, X, Y):
        """
        Generate the transverse refractive potential profile.
        
        For a step-index fiber:
          V(x,y) = k0 * [n(x,y) - n_clad],
        where n(x,y)= n_core if sqrt(x²+y²) ≤ core_radius, and n_clad otherwise.
        """
        if self.fiber_type == 'step-index':
            r = jnp.sqrt(X**2 + Y**2)
            n_xy = radial_index_profile(
                       r,
                       r_core   = self.core_radius,
                       r_clad   = max(self.Lx, self.Ly)/2,   # window radius
                       n_core   = self.n_core,
                       n_clad   = self.n_clad,
                       n_air    = self.n_clad,               # same outside window
                       transition = self.transition,
                       sg_order   = self.sg_order
                   )
            return self.k0 * (n_xy - self.n_clad)
        elif self.fiber_type == 'GRIN':
            if hasattr(self, 'kappa'):
                return 0.5 * self.kappa * (X**2 + Y**2)
            else:
                raise ValueError("GRIN fiber requires attribute 'kappa'.")
        else:
            raise ValueError("Unknown fiber type.")

    def _compute_modes(self, X, Y, wavelength, num_modes=8):
        """Return (modes[Ny,Nx] array, eigenvalues) on the *given* grid."""
        ny, nx = X.shape
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        r  = np.sqrt(X**2 + Y**2)

        n2d = radial_index_profile(
          r,
          r_core   = self.core_radius,
          r_clad   = max(self.Lx, self.Ly)/2,
          n_core   = self.n_core,
          n_clad   = self.n_clad,
          transition = self.transition,
          sg_order   = self.sg_order
      )

        n_flat = n2d.ravel()

        # build sparse Laplacian ⊗ I + I ⊗ Laplacian
        Lx = build_1d_laplacian(nx, dx)
        Ly = build_1d_laplacian(ny, dy)
        Ix = sp.eye(nx)
        Iy = sp.eye(ny)
        Lap = sp.kron(Iy, Lx) + sp.kron(Ly, Ix)

        k0 = 2*np.pi / wavelength
        V  = sp.diags((k0**2) * n_flat**2, 0)

        A = Lap + V            # scalar Helmholtz operator
        eigvals, eigvecs = spla.eigsh(A, k=num_modes, which="LM")

        # sort descending (largest β^2 first)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # reshape + normalise each mode; zero leakage outside core
        modes = []
        rr = r
        for i in range(num_modes):
            m = eigvecs[:, i].reshape((ny, nx))
            m /= np.linalg.norm(m)
            m = np.where(rr <= self.core_radius, m, 0.0)
            m /= np.linalg.norm(m)
            modes.append(m)
        return np.array(modes), eigvals

    # ---------------- public API ---------------- #
    def solve_modes(
        self,
        wavelength: float | None = None,
        num_modes : int = 8,
        cache     : bool = True,
        return_eigvals: bool = False,
        ):
        """
        Solve the scalar Helmholtz eigen‑problem on the current grid and
        return the *raw* modes as a JAX array:
        
            modes  :  (num_modes, Ny, Nx)  with L2‑normalised fields
            betas² :  (num_modes,)         eigen‑values  (optional)
        
        *No automatic LP‑labelling* is attempted – you decide which mode
        indices you want to launch.  (Mode #0 always corresponds to the
        largest β², i.e. LP₀₁ in a weakly‑guiding step‑index fibre.)
        
        To keep disk‑I/O down a small in‑memory cache is still used.
        """
        if wavelength is None:
            wavelength = 2 * np.pi / self.k0
        
        key = (wavelength, num_modes)
        if cache and hasattr(self, "_mode_cache") and key in self._mode_cache:
            modes_arr, eigvals = self._mode_cache[key]
        else:
            # build the same transverse grid as the main simulation
            x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)
            y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)
            X, Y = np.meshgrid(x, y)
        
            modes_np, eigvals = self._compute_modes(X, Y, wavelength, num_modes)
            modes_arr = jnp.asarray(modes_np)        # push to JAX
        
            if cache:
                if not hasattr(self, "_mode_cache"):
                    self._mode_cache = {}
                self._mode_cache[key] = (modes_arr, eigvals)
        
        if return_eigvals:
            return modes_arr, eigvals
        else:
            return modes_arr

    # ---------- quick visual inspection -------------------------------
    def plot_modes(self, modes_arr, cols: int = 4, cmap: str = "RdBu_r"):
        """
        Convenience helper – plot intensity |E|² of the supplied modes.
        """
        n_modes = modes_arr.shape[0]
        rows    = int(np.ceil(n_modes / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.0*cols, 3.0*rows),
                                 sharex=True, sharey=True)
        axes = np.asarray(axes).ravel()
        extent = [-self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2]
    
        for i, ax in enumerate(axes):
            ax.axis("off")
            if i < n_modes:
                im = ax.imshow(np.abs(modes_arr[i])**2, cmap=cmap,
                               origin="lower", extent=extent)
                ax.set_title(f"mode #{i}", fontsize=10)
                ax.axis("on")
        fig.colorbar(im, ax=axes, shrink=0.6, label="|E|² (arb.)")
        fig.suptitle("Solved transverse modes", fontsize=14)
        plt.show()

###############################################################################
# 2. InputSource Class
###############################################################################
class InputSource:
    def __init__(self, power=None, beam_waist=None, pulse_duration=None,
                 source_type="pulsed"):
        """
        `power`, `beam_waist`, `pulse_duration` are ignored for a
        user‑supplied ‘custom’ field but kept for completeness.
        """
        self.power          = power
        self.beam_waist     = beam_waist
        self.pulse_duration = pulse_duration
        self.source_type    = source_type
        self.custom_field   = None            # (Nx,Ny,Nt) – filled later

    # ------------------------------------------------------------------ #
    #  Built‑in launch fields
    # ------------------------------------------------------------------ #
    def pulsed_gaussian(self, x, y, t):
        X, Y, T = jnp.meshgrid(x, y, t, indexing="ij")
        env  = jnp.exp(-(X**2+Y**2)/self.beam_waist**2)
        env *= jnp.exp(-(T**2)/self.pulse_duration**2)
        return jnp.sqrt(self.power) * env

    def cw_gaussian(self, x, y, t):
        X, Y, _ = jnp.meshgrid(x, y, t, indexing="ij")
        env = jnp.exp(-(X**2+Y**2)/self.beam_waist**2)
        return jnp.sqrt(self.power) * env[..., None] * jnp.ones_like(t)

    # ------------------------------------------------------------------ #
    #  NEW: build from modal weights
    # ------------------------------------------------------------------ #
    def modal_superposition(
        self,
        mode_dict,                       # (n_modes,Ny,Nx) array  *or*  {key: field}
        weights: dict,
        phases:  dict | None = None,
        Nt: int = 1,
        *,
        total_power: float | None = None,
        dx: float | None = None,
        dy: float | None = None,
        ):
        """
        Build  A(x,y) = Σ_j w_j e^{iφ_j} · mode_j(x,y)
        
        Parameters
        ----------
        mode_dict
            Either a stacked array  (n_modes, Ny, Nx)  or a mapping
            {key: (Ny,Nx) field}.
        weights
            {key_or_index: amplitude}.
        phases
            {key_or_index: phase [rad]}.  Missing ⇒ 0.
        Nt
            Replicates the transverse field along the temporal axis.
        total_power, dx, dy
            If *total_power* is given the resulting field is rescaled
            so that  ∫∫|A|² dx dy = total_power.
        """
        if phases is None:
            phases = {}
        
        def get_mode(k):
            # supports both styles transparently
            if isinstance(mode_dict, (list, tuple, np.ndarray, jnp.ndarray)):
                return mode_dict[int(k)]
            else:
                return mode_dict[k]
        
        field2d = 0.0
        for key, amp in weights.items():
            phi = phases.get(key, 0.0)
            field2d = field2d + amp * jnp.exp(1j * phi) * get_mode(key)
        
        # ---------- optional normalisation to absolute power -------------
        if total_power is not None:
            if dx is None or dy is None:
                raise ValueError("Need dx and dy to compute power scaling.")
            P_now = jnp.sum(jnp.abs(field2d) ** 2) * dx * dy
            field2d = field2d * jnp.sqrt(total_power / P_now)
        
        # ---------- hook into the rest of the package --------------------
        self.custom_field = jnp.repeat(field2d[..., None], Nt, axis=2)
        self.source_type  = "custom"
        return self.custom_field

    # ------------------------------------------------------------------ #
    #  convenient helper for time‑reversal
    # ------------------------------------------------------------------ #
    @staticmethod
    def time_reverse(field_2d):
        """For the scalar, loss‑less NLSE time‑reversal ⇒ complex conjugate."""
        return jnp.conjugate(field_2d)




###############################################################################
# 3. GNLSE_Sim Class
###############################################################################
class GNLSE_Sim:
    def __init__(self, fiber, input_source, grid_params, time_params=None):
        """
        Set up the simulation.
        
        grid_params must be a dictionary with keys: Nx, Ny, Nt, Lx, Ly, T_win.
        """
        self.fiber = fiber
        self.input_source = input_source
        
        self.Nx = grid_params['Nx']
        self.Ny = grid_params['Ny']
        self.Nt = grid_params['Nt']
        self.Lx = grid_params['Lx']
        self.Ly = grid_params['Ly']
        self.T_win = grid_params['T_win']
        
        self.x = jnp.linspace(-self.Lx/2, self.Lx/2, self.Nx).reshape(-1)
        self.y = jnp.linspace(-self.Ly/2, self.Ly/2, self.Ny).reshape(-1)
        self.t = jnp.linspace(-self.T_win/2, self.T_win/2, self.Nt).reshape(-1)
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')
        
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dt = self.T_win / self.Nt
        
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(self.Nx, d=self.dx)
        self.ky = 2 * jnp.pi * jnp.fft.fftfreq(self.Ny, d=self.dy)
        self.KX, self.KY = jnp.meshgrid(self.kx, self.ky, indexing='ij')
        self.lap_op = -(self.KX**2 + self.KY**2)
        self.omega = 2 * jnp.pi * jnp.fft.fftfreq(self.Nt, d=self.dt)
        
        # Transverse refractive profile from the fiber.
        self.V = self.fiber.refractive_profile(self.X, self.Y)
        self.gamma_space = self.fiber.gamma * jnp.ones((self.Nx, self.Ny))
        
        # Build PML.
        self.pml_thickness = 10
        self.W_x = self.pml_profile_1d(self.x, self.pml_thickness, 1e4)
        self.W_y = self.pml_profile_1d(self.y, self.pml_thickness, 1e4)
        self.W_xy = self.W_x[:, None] + self.W_y[None, :]
        self.W_full = self.W_xy[:, :, None] * jnp.ones((1, 1, self.Nt))
        
        self.args = {
            "Nx": self.Nx,
            "Ny": self.Ny,
            "Nt": self.Nt,
            "beta0": self.fiber.beta0,
            "beta2": self.fiber.beta2,
            "lap_op": self.lap_op,
            "omega": self.omega,
            "V": self.V,
            "gamma": self.gamma_space,
            "W_full": self.W_full,
        }
        
        self.z_samples = None
        self.state_samples = None

    def pml_profile_1d(self, coord, pml_thickness, W_max):
        N = coord.shape[0]
        indices = jnp.arange(N)
        dist = jnp.minimum(indices, N - indices - 1)
        profile = jnp.where(dist < pml_thickness,
                            W_max * ((pml_thickness - dist) / pml_thickness)**2,
                            0.0)
        return profile

    
    def initial_field(self, source_type=None):
        stype = self.input_source.source_type if source_type is None else source_type
        if stype == "pulsed":
            return self.input_source.pulsed_gaussian(self.x, self.y, self.t)
        elif stype == "CW":
            return self.input_source.cw_gaussian(self.x, self.y, self.t)
        elif stype == "custom":
            return self.input_source.custom_field
        else:
            raise ValueError(f"Unknown source type '{stype}'.")


    @staticmethod
    def flatten_field(A):
        A = A.astype(jnp.complex64)
        N_total = A.shape[0] * A.shape[1] * A.shape[2]
        A_re = jnp.real(A).ravel()
        A_im = jnp.imag(A).ravel()
        return jnp.concatenate([A_re, A_im])
    
    @staticmethod
    def reconstruct_field(state, Nx, Ny, Nt):
        N_total = Nx * Ny * Nt
        A_re = state[:N_total].reshape((Nx, Ny, Nt))
        A_im = state[N_total:].reshape((Nx, Ny, Nt))
        return A_re + 1j * A_im

    def gnlse_rhs(self, z, y_flat, args_unused=None):
        """
        The ODE right-hand side in Cartesian form.
        Accepts three arguments as required by diffrax.
        """
        return gnlse_rhs_cartesian(z, y_flat, self.args)
    
    def run_propagation(self, z0, z1, dz_chunk, z_res, n_save_per_chunk, diffrax_solver = "Dopri5"):
        """
        Propagate the initial field from z0 to z1 using chunked integration.
        
        Returns:
          (z_samples, state_samples)
        """
        A0 = self.initial_field()
        y0 = self.flatten_field(A0)
        state = y0
        current_z = z0
        z_samples = []
        state_samples = []

        diffrax_solvers = {"Dopri5": diffrax.Dopri5(), "ReversibleHeun": diffrax.ReversibleHeun()}
        solver = diffrax_solvers[diffrax_solver]
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-4)
        
        tol = 1e-12  # or something a bit larger than your dt
        while current_z + tol < z1:
            step = min(dz_chunk, z1 - current_z)
            local_z = jnp.linspace(current_z, current_z + step, n_save_per_chunk)
            sol_chunk = diffrax.diffeqsolve(
                diffrax.ODETerm(self.gnlse_rhs),
                solver,
                t0=current_z,
                t1=current_z + dz_chunk,
                dt0=step*z_res,
                y0=state,
                args=self.args,
                saveat=diffrax.SaveAt(ts=local_z),
                stepsize_controller=stepsize_controller,
                max_steps=1000000,
                progress_meter = diffrax.TqdmProgressMeter()
            )
            z_samples.extend(np.array(sol_chunk.ts))
            for sol_state in sol_chunk.ys:
                state_samples.append(np.array(sol_state))
            state = sol_chunk.ys[-1]
            current_z += dz_chunk
        
        self.z_samples = np.array(z_samples)
        self.state_samples = state_samples
        return self.z_samples, self.state_samples

    ###############################################################################
    # Visualization Methods
    ###############################################################################
    def plot_transverse_intensity(self, z_index, t_index, fixed_y_index=None):
        """
        Plot the 2D transverse intensity distribution I(x,y) at specified z and t.
        """
        if fixed_y_index is None:
            fixed_y_index = self.Ny // 2
        state = self.state_samples[z_index]
        A = self.reconstruct_field(state, self.Nx, self.Ny, self.Nt)
        I = jnp.abs(A)**2
        plt.figure(figsize=(8,6))
        plt.imshow(np.array(I[:, :, t_index]),
                   extent=[self.y[0], self.y[-1], self.x[0], self.x[-1]],
                   origin='lower', aspect='auto', cmap='viridis')
        plt.xlabel("y (m)")
        plt.ylabel("x (m)")
        plt.title(f"Transverse intensity at z = {self.z_samples[z_index]:.3e} m, t = {self.t[t_index]*1e12:.2f} ps")
        plt.colorbar(label="Intensity (W)")
        plt.show()

    def animate_intensity_z_x_vs_t(self, fixed_y_index=None, skip_frame=1):
        """
        Animate a colormap of intensity I(x,z) as a function of time.
        For each time slice t (sampled with skip_frame), for each saved state, the intensity is
        extracted along x at a fixed y position, then arranged with z on the x-axis and x on the y-axis.
        """
        if fixed_y_index is None:
            fixed_y_index = self.Ny // 2
        
        # Time frame indices.
        t_indices = list(range(0, self.Nt, skip_frame))
        
        # Precompute initial image data.
        intensity_profiles = []
        for state in self.state_samples:
            A = self.reconstruct_field(state, self.Nx, self.Ny, self.Nt)
            I = np.array(jnp.abs(A))
            intensity_profiles.append(I[:, fixed_y_index, t_indices[0]])
        intensity_profiles = np.array(intensity_profiles)  # shape: (num_z, Nx)
        I_data = intensity_profiles.T  # (Nx, num_z)
        
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(I_data, extent=[self.z_samples[0], self.z_samples[-1], self.x[0], self.x[-1]],
                       origin='lower', aspect='auto', cmap='viridis')
        ax.set_xlabel("Propagation distance z (m)")
        ax.set_ylabel("Transverse coordinate x (m)")
        title_text = ax.set_title(f"Intensity at y index = {fixed_y_index}, t = {self.t[t_indices[0]]*1e12:.2f} ps")
        cbar = fig.colorbar(im, ax=ax, label="Intensity (W)")
        
        def update(frame_idx):
            ti = t_indices[frame_idx]
            intensity_profiles = []
            for state in self.state_samples:
                A = self.reconstruct_field(state, self.Nx, self.Ny, self.Nt)
                I = np.array(jnp.abs(A))
                intensity_profiles.append(I[:, fixed_y_index, ti])
            intensity_profiles = np.array(intensity_profiles)
            I_data = intensity_profiles.T
            im.set_data(I_data)
            title_text.set_text(f"Intensity at y index = {fixed_y_index}, t = {self.t[ti]*1e12:.2f} ps")
            return [im, title_text]
        
        ani = animation.FuncAnimation(fig, update, frames=len(t_indices),
                                      interval=150, blit=True)
        plt.close(fig)
        return ani
        
    def animate_intensity_z_x_vs_t_precomp(
            self,
            fixed_y_index=None,
            skip_frame: int = 1,
            cmap: str = "viridis",
            interval: int = 150,
            return_sfl: bool = False,          # <- NEW KEYWORD
            sfl_kind: str = "global_max"       #    optional refinement
        ):
        """
        Animate I(x,z) versus time   ——   *pre‑computes* all images for speed.
    
        Parameters
        ----------
        fixed_y_index : int or None
            y–index to sample. If None uses the fibre centre (Ny//2).
        skip_frame : int
            Use every *skip_frame*‑th temporal grid point.
        cmap : str
            Matplotlib colormap.
        interval : int
            Delay between animation frames [ms].
        return_sfl : bool, optional
            If True the function **also returns** an array `L_sf[t]` that contains
            the self‑focussing length extracted for every displayed t–slice.
        sfl_kind : {"global_max", "first_above_90pc"}, optional
            Metric that defines *L_sf* (see notes below).
    
        Returns
        -------
        ani : matplotlib.animation.FuncAnimation
        L_sf : 1‑D ndarray, only if *return_sfl=True*
            Self‑focussing length for each time‑slice in the animation.
        """
        # ------------------------ set‑up -----------------------------------
        if fixed_y_index is None:
            fixed_y_index = self.Ny // 2
    
        t_frame_indices = list(range(0, self.Nt, skip_frame))
    
        # Pre‑compute intensity images
        precomp_images = []
        # Optional: store L_sf for every t
        sf_lengths = []
    
        for ti in t_frame_indices:
            frame_intensities = []
            for state in self.state_samples:
                A = self.reconstruct_field(state, self.Nx, self.Ny, self.Nt)
                I = np.abs(A)**2                         # intensity
                frame_intensities.append(np.array(I[:, fixed_y_index, ti]))
            frame_image = np.array(frame_intensities).T   # (Nx, n_z)
            precomp_images.append(frame_image)
    
            # --------------   SELF‑FOCUSING LENGTH  -----------------------
            if return_sfl:
                if sfl_kind == "global_max":
                    # z position of absolute maximum in this (x,z) frame
                    col_of_max = np.argmax(frame_image)            # flat index
                    _, z_idx = np.unravel_index(col_of_max, frame_image.shape)
                    sf_lengths.append(self.z_samples[z_idx])
                elif sfl_kind == "first_above_90pc":
                    col_max = frame_image.max()
                    thresh = 0.9*col_max
                    # first z where max_x_intensity crosses 0.9 of the global max
                    max_over_x = frame_image.max(axis=0)
                    z_idx = np.argmax(max_over_x > thresh)
                    sf_lengths.append(self.z_samples[z_idx])
                else:
                    raise ValueError("Unknown sfl_kind option.")
        # ------------------------------------------------------------------
    
        all_intensity = np.concatenate([img.ravel() for img in precomp_images])
        vmin = np.percentile(all_intensity, 1)
        vmax = np.percentile(all_intensity, 99)
    
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(precomp_images[0],
                       extent=[self.z_samples[0], self.z_samples[-1],
                               self.x[0],            self.x[-1]],
                       origin="lower", aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
    
        ax.set_xlabel("Propagation distance  z  (m)")
        ax.set_ylabel("Transverse coordinate  x  (m)")
        title_txt = ax.set_title(
            f"Intensity  (y={fixed_y_index}) at  t = {self.t[t_frame_indices[0]]*1e12:.2f} ps"
        )
        fig.colorbar(im, ax=ax, label="Intensity (W)")
    
        def update(k):
            im.set_data(precomp_images[k])
            title_txt.set_text(
                f"Intensity  (y={fixed_y_index}) at  t = {self.t[t_frame_indices[k]]*1e12:.2f} ps"
            )
            return [im, title_txt]
    
        ani = animation.FuncAnimation(fig, update,
                                      frames=len(t_frame_indices),
                                      interval=interval, blit=True)
        plt.close(fig)
    
        if return_sfl:
            return ani, np.array(sf_lengths)
        else:
            return ani



    def animate_transverse_intensity_vs_z(self, fixed_t_index=None, cmap='viridis', interval=150):
        """
        Animate the full transverse 2D intensity profile I(x,y) at a fixed time slice as a function of z.
        Each frame corresponds to one saved z-snapshot.
        """
        if fixed_t_index is None:
            fixed_t_index = self.Nt // 2
        
        # Precompute initial transverse intensity.
        state0 = self.state_samples[0]
        A0 = self.reconstruct_field(state0, self.Nx, self.Ny, self.Nt)
        I0 = np.array(jnp.abs(A0))[:, :, fixed_t_index]
        
        fig, ax = plt.subplots(figsize=(8,6))
        # Compute extents from the simulation grid.
        extent = [self.y[0], self.y[-1], self.x[0], self.x[-1]]
        im = ax.imshow(I0, extent=extent, origin='lower', cmap=cmap)
        ax.set_xlabel("y (m)")
        ax.set_ylabel("x (m)")
        title_text = ax.set_title(f"Transverse intensity at t index = {fixed_t_index}, z = {self.z_samples[0]:.3e} m")
        cbar = fig.colorbar(im, ax=ax, label="Intensity (W)")
        
        def update(frame_idx):
            state = self.state_samples[frame_idx]
            A = self.reconstruct_field(state, self.Nx, self.Ny, self.Nt)
            I = np.array(jnp.abs(A))[:, :, fixed_t_index]
            im.set_data(I)
            title_text.set_text(f"Transverse intensity at t index = {fixed_t_index}, z = {self.z_samples[frame_idx]:.3e} m")
            return [im, title_text]
        
        ani = animation.FuncAnimation(fig, update, frames=len(self.state_samples),
                                      interval=interval, blit=True)
        plt.close(fig)
        return ani

