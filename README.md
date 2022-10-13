# CENTURI_Tessier_Microglia
Semi-automated detection of microglia and measurement of their dynamics.

## Installation
Download the repository and create a Python environment with the following command line using [Conda](https://docs.conda.io/en/latest/).

    conda env create -f environment.yml

## Usage & content
This package contains two Python scripts. 

The first script, `extract.py`, is used to manually select microglia and process them to investigate their dynamics. This imply multiple rounds of image registration and processing to achieve the segmentation of microglia. Cellular dynamics is evaluated by calculating the percentage of binary pixels whose value changes between two consecutive timepoints. The script produces a folder for each experiment in the repository data folder. These folders contain:
- `coords.csv` - Coordinates (xyz) of the selected microglia
- `crop_reg.tif` - Registered raw images
- `crop_process.tif` - Registered processed images
- `crop_roi.tif` - Cell of interest ROI
- `crop_mask.tif` - Cell of interest masks
- `crop_diff.tif` - Changing binary pixels

The second script, `measure.py`, computes and format results either at the microglia or experiment level. It produces the following files in the repository data folder:
- `all_means.csv` - Merged results for microglia (n=180)
- `all_means_stat.csv` - T test comparing conditions for microglia means  
- `exp_means.csv` - Merged results for experiments (n=11)
- `exp_means_stat.csv` - T test comparing conditions for experiments means  



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


