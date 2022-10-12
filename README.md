# CENTURI_Tessier_Microglia
Semi-automated detection of microglia and measurement of their dynamics.

## Installation
Download the repository and create a Python environment with the following command line using [conda](https://docs.conda.io/en/latest/).

    conda env create -f environment.yml

## Usage
This package contains two Python scripts. The first script, `extract.py`, enables you to manually select microglia and process it to investigate their dynamics. This imply multiple rounds of image registration to minimize movements due to the difficult image acquisition. The second script, `measure.py`, 

## Dependencies
 - Python >= 3.9 
 - [Napari](https://napari.org/stable/)
 - [Numpy](https://numpy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [Scipy](https://scipy.org/)
 - [Scikit-image](https://scikit-image.org/)
 - [Pystackreg](https://pypi.org/project/pystackreg/)
 - [Joblib](https://joblib.readthedocs.io/en/latest/)
 - [Matplotlib](https://matplotlib.org/)


