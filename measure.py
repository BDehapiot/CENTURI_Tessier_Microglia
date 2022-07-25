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

# Get paths
stack_path = Path(Path.cwd(), 'data', stack_name)
fold_path = Path(Path.cwd(), 'data', stack_path.stem)

# Open files
crop_reg = []
crop_mask = []
for path in fold_path.iterdir():
    if 'reg' in path.name:
        path_reg = path
        path_mask = Path(
            fold_path, 
            path.name.replace('reg', 'mask')
            )
        crop_reg.append(io.imread(path_reg))
        crop_mask.append(io.imread(path_mask))
        
#%%

from skimage.filters import sato

ridges = np.zeros_like(crop_reg[0])
for i, frame in enumerate(crop_reg[0]):
    ridges[i,...] = sato(frame.astype('float'), sigmas=1, mode='reflect', black_ridges=False)

#%% Display
# idx = 0
viewer = napari.Viewer()
viewer.add_image(ridges)
# viewer.add_image(crop_mask[idx])


        