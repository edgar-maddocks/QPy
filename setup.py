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
        "pandas",
        "termtables",
        "tqdm",
        "numpy",
        "yfinance",
        "scipy",
        "matplotlib",
        "statsmodels",
        "scikit-learn",
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
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
