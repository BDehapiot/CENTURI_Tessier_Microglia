![Python Badge](https://img.shields.io/badge/Python-3.9-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))  
![Author Badge](https://img.shields.io/badge/Author-Benoit%20Dehapiot-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![Date Badge](https://img.shields.io/badge/Created-2022--07--18-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))     

# CENTURI_Tessier_Microglia  
Semi-automated microglia detection and dynamics  
This code is published here: https://doi.org/10.1093/brain/awad132

## Index
- [Installation](#installation)
- [Usage & content](#usage-&-content)
- [Detailed procedure](#detailed-procedure)
- [Comments](#comments)

## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run the following command:  
```bash
mamba env create -f environment.yml
```
- Activate Conda environment:
```bash
conda activate Microglia
```
Your prompt should now start with `(Microglia)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run the following command: 
```bash
mamba env create -f environment.yml
```  
- Activate Conda environment:  
```bash
conda activate Microglia
```
Your prompt should now start with `(Microglia)` instead of `(base)`

</details>

## Usage & content

This package contains two Python scripts. 

The first script, `extract.py`, is used to manually select microglia and process them to investigate their dynamics. This imply multiple rounds of image registration and processing to achieve the segmentation of microglia. Cellular dynamics is evaluated by calculating the percentage of binary pixels whose value changes between two consecutive timepoints. The script produces a folder for each experiment in the repository data folder. These folders contain:
```bash
- coords.csv          # Coordinates (xyz) of the selected microglia
- crop_reg.tif        # Registered raw images
- crop_process.tif    # Registered processed images
- crop_roi.tif        # Cell of interest ROI
- crop_mask.tif       # Cell of interest masks
- crop_diff.tif       # Changing binary pixels
```
Note that, due to file size limitations on GitHub, we have not included the raw stacks from which the crop data was extracted. However, you can still test the procedure using the subset stack `M1_1d-post-injury_evening_12-05-20_test.tif` present in the repository data folder. Alternatively, the original files can also be made available upon request.

The second script, `measure.py`, computes and format results either at the microglia or experiment level. It produces the following files in the repository data folder:
```bash
- all_means.csv       # Merged results for microglia (n=180)
- all_means_stat.csv  # T test comparing conditions for microglia means  
- exp_means.csv       # Merged results for experiments (n=11)
- exp_means_stat.csv  # T test comparing conditions for experiments means  
```
## Detailed procedure

To compute the microglia cell dynamics we designed a semi-automated procedure using Python. First, the 4D image stacks (3D + time) were loaded into Napari viewer for a manual selection of the cells of interest. Sub-volumes were then cropped around each cell before undergoing a series of spatial registrations using [pystackreg](https://github.com/glichtner/pystackreg) (G. Lichtner, 2022)  ported from the ImageJ [TurboReg](https://doi.org/10.1109/83.650848) plugin (Thevenaz et al, 1998). The aim of these registrations was to correct translationnal, rotational and shearing drift that could be observed during acquisition. After maximum projection along the z-axis, the user was prompted to define regions of interest (ROIs) around each microglia and signal outside of these ROIs was discarded. To segment the microglia, an intensity threshold was determined and applied according to Li's method and the cell dynamics were finally evaluated by calculating the percentage of binary pixels changing in value from one time point to another.

## Comments