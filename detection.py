#%%

import napari
import numpy as np
from skimage import io
from pathlib import Path 
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
rstack = io.imread(rstack_path)

#%%

from skimage.feature import peak_local_max
from skimage.feature import blob_dog
from skimage.feature import blob_log

temp = rstack[0,15,...]

# # local max
# features = peak_local_max(
#     temp, 
#     min_distance=20, 
#     threshold_abs=200, 
#     threshold_rel=None, 
#     exclude_border=True, 
#     indices=True, 
#     footprint=None, 
#     labels=None, 
#     )

# blob dog
features = blob_dog(
    temp,
    min_sigma=1.6, 
    max_sigma=10, 
    sigma_ratio=1.6, 
    threshold=0.1, 
    overlap=0.5,
    threshold_rel=None, 
    exclude_border=False
    )

# # blob log
# features = blob_log(
#     temp,
#     min_sigma=1, 
#     max_sigma=500, 
#     sigma_ratio=1.6, 
#     threshold=50, 
#     overlap=0.5,
#     threshold_rel=None, 
#     exclude_border=False
#     )

#%%

viewer = napari.view_image(temp)

points_layer = viewer.add_points(
    features[:,0:2], 
    size=features[:,2]*10,
    edge_width=0.1,
    edge_color='red',
    face_color='transparent',
    opacity = 0.5,
    )

