# CENTURI_Tessier_Microglia
Semi-automated detection of microglia and measurement of their dynamics.

## Installation
Download the repository and create a Python environment with the following command line using [conda](https://docs.conda.io/en/latest/).

    conda env create -f environment.yml

## Content
This package contains two Python scripts. The first script, `extract.py`, is used to manually select microglia and process them to investigate their dynamics. This imply multiple rounds of image registration and processing to achieve the segmentation of microglia. Cellular dynamics is evaluated by calculating the percentage of binary pixels whose value changes between two consecutive timepoints. 

The second script, `measure.py`, is used to collect and format data from 



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


