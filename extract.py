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

# Inputs
# stack_name = 'M1_1d-post-injury_evening_12-05-20_400x400_8bits.tif'
# stack_name = 'M1_1d-post-injury_evening_12-05-20.tif'
stack_name = 'M2_1d-post-injury_evening_13-05-20.tif'
# stack_name = 'M3_1d-post-injury_morning_13-05-20.tif'
# stack_name = 'M4_1d-post-injury_morning_14-05-20.tif'
# stack_name = 'M6_1d-post-injury_evening_14-05-20.tif'
# stack_name = 'M64_1d-post-injury_evening_23-01-20.tif'
# stack_name = 'M66_1d-post-injury_morning_23-01-20.tif'

xysize = 128
zsize = 5 # should be odd

# Get paths
root_path = Path.cwd()
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
    crop_data : list of ndarray
        Description

    Raises
    ------
    """ 
    
    # Nested function ---------------------------------------------------------
    
    def _imcrop(stack, coord):
        
        sr = StackReg(StackReg.RIGID_BODY)
        tempt = np.zeros([stack.shape[0], xysize, xysize])
        
        z = int(coord[0])
        y = int(coord[1])
        x = int(coord[2])
        
        for t in range(stack.shape[0]):
            
            tempz = stack[t,
                z-zsize//2:z+zsize//2+1,
                y-xysize//2:y+xysize//2,
                x-xysize//2:x+xysize//2,
                ]
            
            # z registration
            tempz = sr.register_transform_stack(tempz, reference='previous')           
            tempt[t,...] = np.max(tempz, axis=0)
        
        # t registration
        tempt = sr.register_transform_stack(tempt, reference='previous')   
        
        return tempt
           
    # Main function -----------------------------------------------------------
        
    crop_data = Parallel(n_jobs=-1)(
        delayed(_imcrop)(
            stack,
            coords[i]
            )
        for i in range(coords.shape[0])
        )

    # crop_data = np.stack([arrays for arrays in output_list], axis=0)

    return crop_data

''' ----------------------------------------------------------------------- '''



#%% Select microglia

# Convert to uint8
stack = range_uint8(stack, int_range=0.999)

# Select microglia
viewer = napari.Viewer()
viewer.add_image(stack[stack.shape[0]//2,...])
points_layer = viewer.add_points(ndim=3)
points_layer.mode = 'add'
_ = show_info('select points')
viewer.show(block=True)
coords = points_layer.data

# Crop data
crop_data = imcrop(stack, coords)

# Define ROI
vertices = []
for i in range(coords.shape[0]):
    viewer = napari.Viewer()
    viewer.add_image(np.mean(crop_data[i], axis=0))
    shapes_layer = viewer.add_shapes(ndim=2)
    shapes_layer.mode = 'add_polygon'
    _ = show_info('select shapes')
    viewer.show(block=True)
    vertices.append(shapes_layer.data[0])
    
crop_data[0].append(1)    

#%% Display
# viewer = napari.Viewer()
# viewer.add_image(crop_data[1])
