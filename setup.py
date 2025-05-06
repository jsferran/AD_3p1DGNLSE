from setuptools import setup, find_packages

setup(
    name="AD_3p1DGNLSE",                    # choose a unique name
    version="0.1.0",
    description="My autodifferentiable 3+1 D GNLSE simulations",
    author="Joseph Ferrantini",
    url="https://github.com/jsferran/AD_3p1DGNLSE",
    packages=find_packages(),            # auto‐discovers the AD_3p1DGNLSE/ folder
    install_requires=[
        "numpy>=1.20",
        "scipy",
        "jax[cuda12]",
        "diffrax"
        # add other dependencies here
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
