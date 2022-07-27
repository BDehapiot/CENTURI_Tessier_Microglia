#%%

import napari
import numpy as np
from skimage import io
from pathlib import Path 
from joblib import Parallel, delayed 

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
thresh_coeff = 1.25

# Get paths
stack_path = Path(Path.cwd(), 'data', stack_name)
dir_path = Path(Path.cwd(), 'data', stack_path.stem)

# Open files
crop_reg = []
crop_roi = []
for path in sorted(dir_path.iterdir()):

    if 'reg' in path.name:
        path_reg = path
        path_roi = Path(
            dir_path, 
            path.name.replace('reg', 'roi')
            )
        crop_reg.append(io.imread(path_reg))
        crop_roi.append(io.imread(path_roi))
        
#%%

from pystackreg import StackReg
from skimage.filters import gaussian, threshold_li
from skimage.morphology import remove_small_holes, remove_small_objects, binary_dilation

sr = StackReg(StackReg.AFFINE)

crop_process = []
crop_mask = []

for reg, roi in zip(crop_reg, crop_roi):
      
    # roi out and blur
    process = reg * gaussian(roi, sigma=5)
    
    # Roll
    ymean = int(np.mean(np.argwhere(roi)[:,0]))
    xmean = int(np.mean(np.argwhere(roi)[:,1]))
    ycorrect = (ymean - xysize//2)*-1
    xcorrect = (xmean - xysize//2)*-1
    process = np.roll(process, (ycorrect, xcorrect), axis=(1, 2))
    
    # t registration
    process = sr.register_transform_stack(process, reference='previous') 
    process[process>1] = 1
    process[process<0] = 0
    
    # Get mask
    mask = process > threshold_li(process) * thresh_coeff
    for t in range(mask.shape[0]):
        
        mask[t,...] = remove_small_objects(
            mask[t,...], min_size=25, connectivity=2
            )
        
        mask[t,...] = remove_small_holes(
            mask[t,...], area_threshold=5, connectivity=2
            )

    # Append list
    crop_process.append(process)
    crop_mask.append(mask)
    
''' ----------------------------------------------------------------------- '''

# crop_std = []

# for process in crop_process:
    
#     std = np.zeros_like(process)
    
#     for t in range(1, std.shape[0]):
        
#         temp_std = process[t-1:t+1]
#         std[t,...] = np.std(temp_std, axis=0) 
        
#     crop_std.append(std)

#%% 

# test = mask.astype('uint8'), binary_dilation(mask.astype('uint8'))
# viewer = napari.Viewer()
# viewer.add_image(test)

#%%

# Clear directory
[files.unlink() for files in dir_path.glob('*process*.tif')] 

# Save data
for i in range(len(crop_process)):
    
    # process
    temp_name = f'crop_process_{i:03}.tif'
    temp_process = crop_process[i] * 255
    temp_process = temp_process.astype('uint8')
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

# idx = 0
# viewer = napari.Viewer()
# viewer.add_image(crop_std[idx])
