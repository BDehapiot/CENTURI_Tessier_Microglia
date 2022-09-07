#%%

import numpy as np
from skimage import io
from pathlib import Path 

#%% Open data

# merged_mask = []
# merged_diff =[]

# data_path = Path(Path.cwd(), 'data')

# for i, fold_path in enumerate(sorted(data_path.iterdir())):   
    
#     if fold_path.is_dir():
        
#         crop_mask = []
#         crop_diff = []
        
#         for path in sorted(fold_path.iterdir()):  
            
#             if 'crop_mask' in path.name:                
#                 crop_mask.append(io.imread(path))
        
#             if 'crop_diff' in path.name:                
#                 crop_diff.append(io.imread(path))     
                
#         merged_mask.append(crop_mask)
#         merged_diff.append(crop_diff)
        
#%%

merged_results = []

data_path = Path(Path.cwd(), 'data')

for exp_path in sorted(data_path.iterdir()):   
    
    if exp_path.is_dir():
        
        for path in sorted(exp_path.iterdir()):  
            
            # Get path and open data           
            if 'crop_mask' in path.name:   
                mask_path = path
                diff_path = Path(str(mask_path).replace('mask', 'diff'))
                mask = io.imread(mask_path)//255
                diff = io.imread(diff_path)
                diff[diff==255] = 0
                diff[diff==127] = 1
                
                # Get results
                areas = np.sum(mask, axis=(1,2))
                sdiff = np.sum(diff, axis=(1,2))
                

            

            #     results = np.sum(mask, axis=(1,2))
                
            # if 'crop_diff' in path.name:                
            #     diff = io.imread(path)  
            #     results = np.concatenate(results, np.sum(diff[diff==127], axis=(1,2)))
                
