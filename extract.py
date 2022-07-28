#%%

import napari
import numpy as np
from skimage import io
from pathlib import Path 
from pystackreg import StackReg
from joblib import Parallel, delayed 
from skimage.draw import polygon2mask
from napari.utils.notifications import show_info
from skimage.filters import gaussian, threshold_li
from skimage.morphology import remove_small_holes, remove_small_objects

#%% To do

# add distance from injury
# get pix & vox size
# 10 to 20 microglia per Z-stack

#%% Get stack name

# stack_name = 'M1_1d-post-injury_evening_12-05-20_400x400_8bits.tif'
# stack_name = 'M1_1d-post-injury_evening_12-05-20.tif'
stack_name = 'M2_1d-post-injury_evening_13-05-20.tif'
# stack_name = 'M3_1d-post-injury_morning_13-05-20.tif'
# stack_name = 'M4_1d-post-injury_morning_14-05-20.tif'
# stack_name = 'M6_1d-post-injury_evening_14-05-20.tif'
# stack_name = 'M64_1d-post-injury_evening_23-01-20.tif'
# stack_name = 'M66_1d-post-injury_morning_23-01-20.tif'

#%% Initialize

# Parameters
preload = False # Load a preselected coordinates
xysize = 128
zsize = 5
thresh_coeff = 1.25

# Get paths
stack_path = Path(Path.cwd(), 'data', stack_name)

# Create directory
dir_path = Path(Path.cwd(), 'data', stack_path.stem)
dir_path.mkdir(exist_ok=True)

# Clear directory
if not preload:
    [files.unlink() for files in dir_path.glob("*")] 

# Open file
stack = io.imread(stack_path)

#%% functions

def range_uint8(img, int_range=0.99):

    ''' 
    Description
    
    Parameters
    ----------
    img : ndarray
        Description        
        
    int_range : float
        Description
    
    Returns
    -------  
    img : ndarray
        Description
        
    Notes
    -----   
    
    '''

    # Get data type 
    data_type = (img.dtype).name
    
    if data_type == 'uint8':
        
        raise ValueError('Input image is already uint8') 
        
    else:
        
        # Get data intensity range
        int_min = np.percentile(img, (1-int_range)*100)
        int_max = np.percentile(img, int_range*100) 
        
        # Rescale data
        img[img<int_min] = int_min
        img[img>int_max] = int_max 
        img = (img - int_min)/(int_max - int_min)
        img = (img*255).astype('uint8')
    
    return img

''' ----------------------------------------------------------------------- '''

def imselect(stack, preload=preload):
    
    """
    Description
    
    Parameters
    ----------
    stack : ndarray
        Description
                
    preload : bool
        Description
    
    Returns
    -------
    coords : ndarray
        Description

    Raises
    ------
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

''' ----------------------------------------------------------------------- '''

def imreg(stack, coords, xysize=xysize, zsize=zsize, preload=preload):
    
    """
    Description
    
    Parameters
    ----------
    stack : ndarray
        Description
        
    coords : ndarray
        Description
        
    xysize : int
        Description
        
    zsize : int
        Description
        
    preload : bool
        Description
    
    Returns
    -------
    crop_reg : list of ndarray
        Description

    Raises
    ------
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
                crop_reg.append(io.imread(path))
        
    else:
    
        crop_reg = Parallel(n_jobs=-1)(
            delayed(_imreg)(
                stack,
                coords[i]
                )
            for i in range(coords.shape[0])
            )

    return crop_reg

''' ----------------------------------------------------------------------- '''

def imroi(crop_reg, coords, xysize=xysize, preload=preload):
    
    """
    Description
    
    Parameters
    ----------
    crop_reg : list of ndarray
        Description
        
    coords : ndarray
        Description
        
    preload : bool
        Description
    
    Returns
    -------
    crop_roi : list of ndarray
        Description

    Raises
    ------
    """ 
    
    if preload:
        
        crop_roi = []
        for path in sorted(dir_path.iterdir()):           
            if 'roi' in path.name:
                crop_roi.append(io.imread(path))
    
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

''' ----------------------------------------------------------------------- '''

def improcess(crop_reg, crop_roi):
    
    """
    Description
    
    Parameters
    ----------
    crop_reg : list of ndarray
        Description
        
    crop_roi : list of ndarray
        Description

    Returns
    -------
    crop_process : list of ndarray
        Description

    Raises
    ------
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

''' ----------------------------------------------------------------------- '''

def immask(crop_process, thresh_coeff=thresh_coeff):
    
    """
    Description
    
    Parameters
    ----------
    crop_process : list of ndarray
        Description
        
    thresh_coeff : float
        Description
    
    Returns
    -------
    crop_mask : list of ndarray
        Description

    Raises
    ------
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

''' ----------------------------------------------------------------------- '''

def imdiff(crop_mask):
    
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
stack = range_uint8(stack, int_range=0.999)
# Select 
coords = imselect(stack)
# Crop & register
crop_reg = imreg(stack, coords)
# Draw ROIs
crop_roi = imroi(crop_reg, coords)
# Process
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
            check_contrast=False
            )
        
        # roi
        temp_name = f'crop_roi_{i:03}.tif'
        io.imsave(
            Path(dir_path, temp_name),
            crop_roi[i].astype('uint8'),
            check_contrast=False
            )
    
    # process
    temp_name = f'crop_process_{i:03}.tif'
    temp_process = crop_process[i].astype('uint8')
    io.imsave(
        Path(dir_path, temp_name),
        temp_process,
        check_contrast=False
        )
    
    # mask
    temp_name = f'crop_mask_{i:03}.tif'
    temp_mask = crop_mask[i] * 255
    temp_mask = temp_mask.astype('uint8')
    io.imsave(
        Path(dir_path, temp_name),
        temp_mask,
        check_contrast=False
        )

#%% Display

idx = 0
viewer = napari.Viewer()
viewer.add_image(crop_process[idx])
viewer.add_image(crop_mask[idx])
viewer.add_image(crop_diff[idx]*255)
viewer.grid.enabled = True