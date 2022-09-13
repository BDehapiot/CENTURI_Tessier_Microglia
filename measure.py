#%%

import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path 

#%%

all_data = []
all_means = []

data_path = Path(Path.cwd(), 'data')

for exp_path in sorted(data_path.iterdir()):   
    
    if exp_path.is_dir():
        
        idx = -1
        
        for path in sorted(exp_path.iterdir()):  
                        
            # Get path and open data           
            if 'crop_mask' in path.name:  
                idx = idx + 1 # get cell id
                mask_path = path
                diff_path = Path(str(mask_path).replace('mask', 'diff'))
                mask = io.imread(mask_path)//255
                diff = io.imread(diff_path)
                diff[diff==255] = 0
                diff[diff==127] = 1
                
                # Get data & means 
                area = np.sum(mask, axis=(1,2)) # mask area (pixels)
                ndiff = np.sum(diff, axis=(1,2))/area # % of changed pixels
                area_mean = np.mean(area)
                area_sd = np.std(area)
                ndiff_mean = np.mean(ndiff)
                ndiff_sd = np.std(ndiff)
                
                # Append data 
                all_data.append((
                    exp_path.stem, # name of the experiment
                    idx, # cell_id
                    area, ndiff # data
                    ))
                
                # Append means
                all_means.append((
                    exp_path.stem, # name of the experiment
                    idx, # cell_id
                    area_mean, area_sd, 
                    ndiff_mean, ndiff_sd  
                    ))                

#%%

all_data = pd.DataFrame(all_data)
all_data.columns = [
    'exp_name', 
    'cell_id', 
    'area', 
    'change_ratio',
    ]

all_means = pd.DataFrame(all_means)
all_means.columns = [
    'exp_name', 
    'cell_id', 
    'area_mean',
    'area_sd',
    'change_ratio_mean',
    'change_ratio_sd'
    ]

#%%

exp_means = []

exp_names = all_means.exp_name.unique()

for exp_name in exp_names:
    
    temp = all_means[all_means['exp_name'] == exp_name]
    
    exp_means.append((
        exp_name,
        np.mean(temp['area_mean']),
        np.mean(temp['area_sd']),
        np.mean(temp['change_ratio_mean']),
        np.mean(temp['change_ratio_sd']),
        ))

exp_means = pd.DataFrame(exp_means)
exp_means.columns = [
    'exp_name', 
    'area_mean',
    'area_sd',
    'change_ratio_mean',
    'change_ratio_sd'
    ]    
    

# test = all_means[all_means['exp. name'] == 'M1_1d-post-injury_evening_12-05-20']               
