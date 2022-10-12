#%% Imports 

import napari
import numpy as np
from skimage import io
from pathlib import Path 
from tifffile import imread
from pystackreg import StackReg
from joblib import Parallel, delayed 
from skimage.draw import polygon2mask
from napari.utils.notifications import show_info
from skimage.filters import gaussian, threshold_li
from skimage.morphology import remove_small_holes, remove_small_objects

#%% Parameters

preload = False # select microglia using saved coordinates
xysize = 128 # size of the crop region (pixels)
zsize = 5 # depth of the crop region, should be odd (slices)
thresh_coeff = 2 # adjust segmentation threshold using this coefficient

#%% Get stack name

# Test stack (included in GitHub repository)
stack_name = 'M1_1d-post-injury_evening_12-05-20_test.tif'
 
# stack_name = 'M1_1d-post-injury_evening_12-05-20.tif'
# stack_name = 'M2_1d-post-injury_evening_13-05-20.tif'
# stack_name = 'M4_1d-post-injury_morning_14-05-20.tif'
# stack_name = 'M6_1d-post-injury_evening_14-05-20.tif'
# stack_name = 'M64_1d-post-injury_evening_23-01-20.tif'
# stack_name = 'M66_1d-post-injury_morning_23-01-20.tif'
# stack_name = 'M1_5d-post-injury_evening_16-05-20.tif'
# stack_name = 'M2_5d-post-injury_evening_17-05-20.tif'
# stack_name = 'M3_5d-post-injury_morning_17-05-20.tif'
# stack_name = 'M4_5d-post-injury_morning_18-05-20.tif'
# stack_name = 'M6_5d-post-injury_evening_18-05-20.tif'

#%% Initialize

# Get paths
stack_path = Path(Path.cwd(), 'data', stack_name)

# Create directory
dir_path = Path(Path.cwd(), 'data', stack_path.stem)
dir_path.mkdir(exist_ok=True)

# Clear directory
if not preload:
    [files.unlink() for files in dir_path.glob("*")] 

# Open file
stack = imread(stack_path)

#%% functions

def ranged_uint8(img, percent_low=1, percent_high=99):

    """ 
    Convert image to uint8 using a percentile range.
    
    Parameters
    ----------
    img : ndarray
        Image to be converted.
        
    percent_low : float
        Percentile to discard low values.
        Between 0 and 100 inclusive.
        
    percent_high : float
        Percentile to discard high values.
        Between 0 and 100 inclusive.
    
    Returns
    -------  
    img : ndarray
        Converted image.
    
    """

    # Get data type 
    data_type = (img.dtype).name
    
    if data_type == 'uint8':
        
        raise ValueError('Input image is already uint8') 
        
    else:
        
        # Get data intensity range
        int_min = np.percentile(img, percent_low)
        int_max = np.percentile(img, percent_high) 
        
        # Rescale data
        img[img<int_min] = int_min
        img[img>int_max] = int_max 
        img = (img - int_min)/(int_max - int_min)
        img = (img*255).astype('uint8')
    
    return img

""" ----------------------------------------------------------------------- """

def imselect(stack, preload=preload):
    
    """ 
    Get xyz coordinates of cells of interest (COIs). 
    
    Once Napari is open, select COIs and close the window to proceed.
    If preload, the code will instead import previously saved coordinates.
    
    Parameters
    ----------
    stack : ndarray
        Source image stack. 
                
    preload : bool
        Load previously saved coordinates.
    
    Returns
    -------
    coords : ndarray
        Cells of interest coordinates.

    """ 
    
    if preload and Path(dir_path, 'coords.csv').is_file():
        
        # Load preselected coordinates
        coords = np.genfromtxt(Path(dir_path, 'coords.csv'), delimiter=',')
        
    else:
    
        # Display in Napari
        viewer = napari.Viewer()
        viewer.add_image(stack[stack.shape[0]//2,...]) # mid frame
        
        # Extract coordinates
        points_layer = viewer.add_points(ndim=3)
        points_layer.mode = 'add'
        _ = show_info('select points')
        viewer.show(block=True)
        coords = points_layer.data

    return coords

""" ----------------------------------------------------------------------- """

def imreg(stack, coords, xysize=xysize, zsize=zsize, preload=preload):
    
    """ Crop, register and project cells of interest. 
    
    The function first extract a cropped region of xysize (pixels) by zsize 
    (slices) centered around the imported coordinates. The cropped region is
    then registered in xy through z axis for every timepoints. Finally, The z 
    volume is max-projected and registered in xy over time. 
    
    If preload, the code will instead import previously saved images.
    
    Parameters
    ----------
    stack : ndarray
        Source image stack.
        
    coords : ndarray
        Cells of interest coordinates.
        
    xysize : int
        Size of the crop region (pixels)
        
    zsize : int
        Depth of the crop region, should be odd (slices)
        
    preload : bool
        Load previously saved images.
    
    Returns
    -------
    crop_reg : list of ndarray
        Cropped and registered images.

    """ 
    
    # Nested function ---------------------------------------------------------
    
    def _imreg(stack, coord, preload=preload):
        
        sr = StackReg(StackReg.RIGID_BODY)
        tempt = np.zeros([stack.shape[0], xysize, xysize])
        
        # Define z
        z = int(coord[0])
        zi = z-zsize//2; zf = z+zsize//2+1
        
        if zi < 0:
            zi = 0; zf = zsize
                                   
        if zf > stack.shape[1]:
            zi = stack.shape[1]-(zsize)
            zf = stack.shape[1]

        # Define y
        y = int(coord[1])
        yi = y-xysize//2; yf = y+xysize//2
        
        if yi < 0:
            yi = 0; yf = xysize
        
        if yf > stack.shape[2]:
            yi = stack.shape[2]-(xysize)
            yf = stack.shape[2]
            
        # Define x
        x = int(coord[2])
        xi = x-xysize//2; xf = x+xysize//2
        
        if xi < 0:
            xi = 0; xf = xysize
        
        if xf > stack.shape[3]:
            xi = stack.shape[3]-(xysize)
            xf = stack.shape[3]
        
        for t in range(stack.shape[0]):
           
            # Extract cropped images
            tempz = stack[t, zi:zf, yi:yf, xi:xf]
            
            # z registration
            tempz = sr.register_transform_stack(tempz, reference='previous')           
            tempt[t,...] = np.max(tempz, axis=0)
        
        # t registration
        tempt = sr.register_transform_stack(tempt, reference='previous') 
        
        return tempt
           
    # Main function -----------------------------------------------------------
        
    if preload:
        
        crop_reg = []
        for path in sorted(dir_path.iterdir()):           
            if 'reg' in path.name:
                crop_reg.append(imread(path))
        
    else:
    
        crop_reg = Parallel(n_jobs=-1)(
            delayed(_imreg)(
                stack,
                coords[i]
                )
            for i in range(coords.shape[0])
            )

    return crop_reg

""" ----------------------------------------------------------------------- """

def imroi(crop_reg, coords, xysize=xysize, preload=preload):
    
    """
    Get cell of interest ROI.
    
    Once Napari is open, draw a polygon around the cell of interest and close
    the window to move on to the next.
    
    If preload, the code will instead import previously saved ROIs.    
    
    Parameters
    ----------
    crop_reg : list of ndarray
        Cropped and registered images.
        
    coords : ndarray
        Cells of interest coordinates.
        
    preload : bool
        Load previously saved ROI.
    
    Returns
    -------
    crop_roi : list of ndarray
        ROI images.

    Raises
    ------
    """ 
    
    if preload:
        
        crop_roi = []
        for path in sorted(dir_path.iterdir()):           
            if 'roi' in path.name:
                crop_roi.append(imread(path))
    
    else:
    
        crop_roi = []
            
        for i in range(coords.shape[0]):
            
            # Display in Napari
            viewer = napari.Viewer()
            viewer.add_image(np.mean(crop_reg[i], axis=0)) # mean temporal proj.
            
            # Extract vertices
            shapes_layer = viewer.add_shapes(ndim=2)
            shapes_layer.mode = 'add_polygon'
            _ = show_info('select shapes')
            viewer.show(block=True)
            vertices = shapes_layer.data[0]
            
            # Append crop_roi
            crop_roi.append(polygon2mask((xysize, xysize), vertices))

    return crop_roi

""" ----------------------------------------------------------------------- """

def improcess(crop_reg, crop_roi):
    
    """
    Process images for binarization.
    
    The function first remove out of ROI signals and then center the cell of 
    interest. Finally, an affine registration is applied to reduce shear 
    artifacts due to acquisition conditions. 
    
    Parameters
    ----------
    crop_reg : list of ndarray
        Cropped and registered images.
        
    crop_roi : list of ndarray
        ROI images.

    Returns
    -------
    crop_process : list of ndarray
        Processed images.

    """ 
    sr = StackReg(StackReg.AFFINE)

    crop_process = []
    
    for reg, roi in zip(crop_reg, crop_roi):
          
        # Remove out of roi signal
        process = reg * gaussian(roi, sigma=5)
        
        # Roll image
        ymean = int(np.mean(np.argwhere(roi)[:,0]))
        xmean = int(np.mean(np.argwhere(roi)[:,1]))
        ycorrect = (ymean - xysize//2)*-1
        xcorrect = (xmean - xysize//2)*-1
        process = np.roll(process, (ycorrect, xcorrect), axis=(1, 2))
        
        # t registration
        process = sr.register_transform_stack(process, reference='previous') 
        process[process>255] = 255
        process[process<0] = 0

        # Append crop_process
        crop_process.append(process)
    
    return crop_process

""" ----------------------------------------------------------------------- """

def immask(crop_process, thresh_coeff=thresh_coeff):
    
    """
    Segment cells of interest using Li thresholding. 
    
    Threshold can be adjusted using the thresh_coeff parameter.
    
    Parameters
    ----------
    crop_process : list of ndarray
        Processed images.
        
    thresh_coeff : float
        Adjust the threshold value.
        < 1 expand the segmented area.
        > 1 reduce the segmented area.
    
    Returns
    -------
    crop_mask : list of ndarray
        Binarized images.
        
    """ 
    
    crop_mask = []
    
    for process in crop_process:
        
        # Get mask
        mask = process > threshold_li(process) * thresh_coeff
        
        for t in range(mask.shape[0]):
            
            mask[t,...] = remove_small_objects(
                mask[t,...], min_size=25, connectivity=2
                )
            
            mask[t,...] = remove_small_holes(
                mask[t,...], area_threshold=5, connectivity=2
                )        
        
        # Append crop_mask
        crop_mask.append(mask)
    
    return crop_mask

""" ----------------------------------------------------------------------- """

def imdiff(crop_mask):
    
    """
    Evaluate cells of interest dynamics. 
    
    Average pixel value from t-1 to t0. 
    Static pixels = 0 or 1, while changing pixels = 0.5. 
    
    Note that first timepoint is not evaluated due to the lack of t-1. 
        
    Parameters
    ----------
    crop_mask : list of ndarray
        Binarized images.

    Returns
    -------
    crop_diff : list of ndarray
        Cell dynamics images.    
    
    """ 
    
    crop_diff = []
    
    for mask in crop_mask:  
        
        mask = mask.astype('float')
        diff = np.zeros_like(mask)
        
        for t in range(1, mask.shape[0]):
            
            diff[t,...] = np.mean(mask[t-1:t+1,...], axis=0)
            
        crop_diff.append(diff)      
    
    return crop_diff

#%% Main

# Convert to uint8
stack = ranged_uint8(stack, percent_low=0.1, percent_high=99.9)
# Select 
coords = imselect(stack, preload=preload)
# Crop & register
crop_reg = imreg(stack, coords, preload=preload)
# Draw ROIs
crop_roi = imroi(crop_reg, coords, preload=preload)
# Process-
crop_process = improcess(crop_reg, crop_roi)
# Get mask
crop_mask = immask(crop_process)
# Get diff
crop_diff = imdiff(crop_mask)

#%% Save

for i in range(coords.shape[0]):

    # coords
    np.savetxt(Path(dir_path, 'coords.csv'), coords, delimiter=',')

    if not preload:
        
        # reg
        temp_name = f'crop_reg_{i:03}.tif'
        temp_reg = crop_reg[i]
        temp_reg[temp_reg > 255] = 255
        temp_reg[temp_reg < 0] = 0
        temp_reg = temp_reg.astype('uint8')
        io.imsave(
            Path(dir_path, temp_name),
            temp_reg,
            check_contrast=False,
            imagej=True,
            metadata={'axes': 'ZYX'}
            )
        
        # roi
        temp_name = f'crop_roi_{i:03}.tif'
        io.imsave(
            Path(dir_path, temp_name),
            crop_roi[i].astype('uint8'),
            check_contrast=False,
            )
    
    # process
    temp_name = f'crop_process_{i:03}.tif'
    temp_process = crop_process[i]
    temp_process[temp_process > 255] = 255
    temp_process[temp_process < 0] = 0
    temp_process = temp_process.astype('uint8')
    io.imsave(
        Path(dir_path, temp_name),
        temp_process,
        check_contrast=False,
        imagej=True,
        metadata={'axes': 'ZYX'}
        )
    
    # mask
    temp_name = f'crop_mask_{i:03}.tif'
    temp_mask = crop_mask[i] * 255
    temp_mask = temp_mask.astype('uint8')
    io.imsave(
        Path(dir_path, temp_name),
        temp_mask,
        check_contrast=False,
        imagej=True,
        metadata={'axes': 'ZYX'}
        )
    
    # diff
    temp_name = f'crop_diff_{i:03}.tif'
    temp_diff = crop_diff[i] * 255
    temp_diff = temp_diff.astype('uint8')
    io.imsave(
        Path(dir_path, temp_name),
        temp_diff,
        check_contrast=False,
        imagej=True,
        metadata={'axes': 'ZYX'}
        )

#%% Display

# idx = 0
# viewer = napari.Viewer()
# viewer.add_image(crop_process[idx])
# viewer.add_image(crop_mask[idx])
# viewer.add_image(crop_diff[idx]*255)
# viewer.grid.enabled = True