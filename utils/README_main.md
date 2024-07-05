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