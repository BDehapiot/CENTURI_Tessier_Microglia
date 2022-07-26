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

# Get paths
stack_path = Path(Path.cwd(), 'data', stack_name)
dir_path = Path(Path.cwd(), 'data', stack_path.stem)

# Open files
crop_reg = []
crop_mask = []
for path in sorted(dir_path.iterdir()):

    if 'reg' in path.name:
        path_reg = path
        path_mask = Path(
            dir_path, 
            path.name.replace('reg', 'mask')
            )
        crop_reg.append(io.imread(path_reg))
        crop_mask.append(io.imread(path_mask))
        
#%%

from pystackreg import StackReg
from skimage.filters import gaussian

sr = StackReg(StackReg.AFFINE)

crop_process = []

for reg, mask in zip(crop_reg, crop_mask):
      
    # Mask out and blur
    process = reg * gaussian(mask, sigma=5)
    process = gaussian(process, sigma=0.5)   
    
    # Roll
    ymean = int(np.mean(np.argwhere(mask)[:,0]))
    xmean = int(np.mean(np.argwhere(mask)[:,1]))
    ycorrect = (ymean - xysize//2)*-1
    xcorrect = (xmean - xysize//2)*-1
    process = np.roll(process, (ycorrect, xcorrect), axis=(1, 2))
    
    # t registration
    process = sr.register_transform_stack(process, reference='previous') 
    
    # Append crop_process
    crop_process.append(process)
    
''' ----------------------------------------------------------------------- '''

crop_std = []

for process in crop_process:
    
    std = np.zeros_like(process)
    
    for t in range(1, std.shape[0]):
        
        temp_std = process[t-1:t+1]
        std[t,...] = np.std(temp_std, axis=0) 
        
    crop_std.append(std)
    
    # Denoising
    frame_movie = cv2.fastNlMeansDenoising(frame_movie)
    plt.imshow(frame_movie, cmap=cm.Greys)
    plt.title(f"After denoising")
    plt.show()
    
    # Gaussian Local Thresholding
    binary_frame = cv2.adaptiveThreshold(
        frame_movie, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

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
    
#%% Display

idx = 0
viewer = napari.Viewer()
viewer.add_image(crop_std[idx])

#%%
