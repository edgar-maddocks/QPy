from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "QPy - Quant finance library"
LONG_DESCRIPTION = "An all-in-one quantitative finance library which aims to implement ML and traditional quantitative methods"

# Setting up
setup(
    name="QPy",
    version=VERSION,
    author="Edgar Maddocks",
    author_email="edgarmaddocks@outlook.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "pandas>=2.1.0",
        "termtables>=0.2.4",
        "tqdm>=4.66.1",
        "numpy>=1.25.2",
        "yfinance>=0.2.28",
        "scipy>=1.11.2",
        "matplotlib>=3.7.2",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.2",
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=[
        "python",
        "first package",
        "quant",
        "quantitative",
        "finance",
        "backtesting",
        "portfolio",
        "stats",
    ],
)
