#%%

import napari
import numpy as np
from skimage import io
from pathlib import Path 
from skimage.draw import polygon2mask
from pystackreg import StackReg
from joblib import Parallel, delayed 
from napari.utils.notifications import show_info

#%%

# stack_name = 'M1_1d-post-injury_evening_12-05-20_400x400_8bits.tif'
# stack_name = 'M1_1d-post-injury_evening_12-05-20.tif'
stack_name = 'M2_1d-post-injury_evening_13-05-20.tif'
# stack_name = 'M3_1d-post-injury_morning_13-05-20.tif'
# stack_name = 'M4_1d-post-injury_morning_14-05-20.tif'
# stack_name = 'M6_1d-post-injury_evening_14-05-20.tif'
# stack_name = 'M64_1d-post-injury_evening_23-01-20.tif'
# stack_name = 'M66_1d-post-injury_morning_23-01-20.tif'

# Inputs
xysize = 128
zsize = 5 # should be odd

# Get paths
stack_path = Path(Path.cwd(), 'data', stack_name)

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

def imselect(stack):
    
    """
    Description
    
    Parameters
    ----------
    stack : ndarray
        Description
    
    Returns
    -------
    coords : ndarray
        Description

    Raises
    ------
    """ 
    
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

def imcrop(stack, coords, xysize=xysize, zsize=zsize):
    
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
    
    Returns
    -------
    crop_reg : list of ndarray
        Description

    Raises
    ------
    """ 
    
    # Nested function ---------------------------------------------------------
    
    def _imcrop(stack, coord):
        
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
        
    crop_reg = Parallel(n_jobs=-1)(
        delayed(_imcrop)(
            stack,
            coords[i]
            )
        for i in range(coords.shape[0])
        )

    return crop_reg

''' ----------------------------------------------------------------------- '''

def immask(crop_reg, coords, xysize=xysize):
    
    """
    Description
    
    Parameters
    ----------
    crop_reg : list of ndarray
        Description
        
    coords : ndarray
        Description
    
    Returns
    -------
    crop_mask : list of ndarray
        Description

    Raises
    ------
    """ 
    
    crop_mask = []
        
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
        
        # Extract vertices
        crop_mask.append(polygon2mask((xysize, xysize), vertices))

    return crop_mask

#%% Main

# Convert to uint8
stack = range_uint8(stack, int_range=0.999)
# Select
coords = imselect(stack)
# Crop
crop_reg = imcrop(stack, coords)
# Mask
crop_mask = immask(crop_reg, coords)

#%% Save

# Create directory
dir_path = Path(Path.cwd(), 'data', stack_path.stem)
dir_path.mkdir(exist_ok=True)

# Clear directory
[files.unlink() for files in dir_path.glob("*")] 

# Save data
for i in range(coords.shape[0]):
    
    # reg
    temp_name = f'crop_reg_{i:03}.tif'
    temp_raw = crop_reg[i]
    temp_raw[temp_raw > 255] = 255
    temp_raw[temp_raw < 0] = 0
    temp_raw = temp_raw.astype('uint8')
    io.imsave(
        Path(dir_path, temp_name),
        temp_raw,
        check_contrast=False
        )
    
    # mask
    temp_name = f'crop_mask_{i:03}.tif'
    io.imsave(
        Path(dir_path, temp_name),
        crop_mask[i].astype('uint8'),
        check_contrast=False
        )

#%% Display
# idx = 0
# viewer = napari.Viewer()
# viewer.add_image(crop_reg[idx])
# viewer.add_image(crop_mask[idx])