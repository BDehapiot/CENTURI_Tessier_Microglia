# CENTURI_Tessier_Microglia
Semi-automated detection of microglia and measurement of their dynamics.

## Installation
Download the repository and create a Python environment with the following command line using [Conda](https://docs.conda.io/en/latest/).

    conda env create -f environment.yml

## Content
This package contains two Python scripts. The first script, `extract.py`, is used to manually select microglia and process them to investigate their dynamics. This imply multiple rounds of image registration and processing to achieve the segmentation of microglia. Cellular dynamics is evaluated by calculating the percentage of binary pixels whose value changes between two consecutive timepoints. The `extract.py` script produces a folder for each experiment in the repository data folder. These folders contain:
- `coords.csv` (xyz coordinates of the selected microglia)    
- `crop_reg.tif` (registered images)
- `crop_roi.tif` (cell of interest ROI)
- `crop_process.tif` ()
- `crop_mask.tif` ()
- `crop_diff.tif` ()

The second script, `measure.py`, is used to format data and get results 



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


