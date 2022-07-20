#%%

import napari
import numpy as np
from skimage import io
from pathlib import Path 
from pystackreg import StackReg
from joblib import Parallel, delayed 

#%%

# root = '/media/bdehapiot/DATA/CurrentTasks/CENTURIProject_INMED_MarineTessier/'
# name = 'M1_1d-post-injury_evening_12-05-20.tif'
# stack = io.imread(root + name)

# Inputs
stack_name = 'M1_1d-post-injury_evening_12-05-20_400x400_8bits.tif'

# Get paths
data_path = Path(Path.cwd() / 'data' )
stack_path = Path(Path.cwd() / 'data' / stack_name)
rstack_path = Path(Path.cwd() / 'data' / (stack_path.stem + '_reg.tif'))

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

def imreg(stack, zreg=True, treg=True, parallel=True):
    
    """
    Description
    
    Parameters
    ----------
    stack : ndarray
        Description
    
    Returns
    -------
    rstack : ndarray
        Description

    Raises
    ------
    """ 
    
    # Nested function ---------------------------------------------------------
       
    def _imreg_z(zstack):
        
        zstack = sr.register_transform_stack(
            zstack, reference='previous')        

        return zstack
    
    def _imreg_t(tstack):
        
        tstack = sr.transform_stack(tstack)      

        return tstack
           
    # Main function -----------------------------------------------------------
    
    rstack = stack.copy()
    
    if zreg:
    
        # Register stack in z
        sr = StackReg(StackReg.RIGID_BODY)
        
        if parallel:
     
            output_list = Parallel(n_jobs=-1)(
                delayed(_imreg_z)(
                    stack[t,...],
                    )
                for t in range(stack.shape[0])
                )
                
        else:
    
            output_list = [_imreg_z(
                    stack[t,...],
                    ) 
                for t in range(stack.shape[0])
                ]
    
        rstack = np.stack([arrays for arrays in output_list], axis=0)
    
    if treg:
    
        # Register stack in t
        sr = StackReg(StackReg.RIGID_BODY)
        sr.register_stack(np.max(rstack, axis=1), reference='previous')
        
        if parallel:
            
            output_list = Parallel(n_jobs=-1)(
                delayed(_imreg_t)(
                    rstack[:,z,...],
                    )
                for z in range(rstack.shape[1])
                )
            
        else:
            
            output_list = [_imreg_t(
                    rstack[:,z,...],
                    ) 
                for z in range(rstack.shape[1])
                ]
    
        rstack = np.stack([arrays for arrays in output_list], axis=0)
        rstack = np.swapaxes(rstack,0,1)

    return rstack

#%%

rstack = imreg(stack, zreg=True, treg=True)

io.imsave(
    rstack_path, 
    range_uint8(rstack, int_range=0.999), 
    check_contrast=False, 
    imagej=True, 
    metadata={'axes': 'TZYX'},
    )

#%%

viewer = napari.view_image(rstack)