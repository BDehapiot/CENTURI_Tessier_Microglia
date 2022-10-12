#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path 
from tifffile import imread
import matplotlib.pyplot as plt
from scipy import stats

#%% Get folder name

ctrl_1d = [
    'M1_1d-post-injury_evening_12-05-20',
    'M2_1d-post-injury_evening_13-05-20', 
    'M64_1d-post-injury_evening_23-01-20'
    ]

ctrl_5d = [
    'M1_5d-post-injury_evening_16-05-20',
    'M2_5d-post-injury_evening_17-05-20'
    ]

bum_1d = [
    'M4_1d-post-injury_morning_14-05-20',
    'M6_1d-post-injury_evening_14-05-20',
    'M66_1d-post-injury_morning_23-01-20'    
    ]

bum_5d = [
    'M3_5d-post-injury_morning_17-05-20',
    'M4_5d-post-injury_morning_18-05-20',
    'M6_5d-post-injury_evening_18-05-20'
    ]

#%% Extract data & get results

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
                mask = imread(mask_path)//255
                diff = imread(diff_path)               
                diff[diff==255] = 0
                diff[diff==127] = 1
                                
                # get exp. and cond. names
                exp_name = exp_path.stem
                if exp_name in ctrl_1d:
                    cond_name = 'ctrl_1d' 
                if exp_name in ctrl_5d:
                    cond_name = 'ctrl_5d'
                if exp_name in bum_1d:
                    cond_name = 'bum_1d'
                if exp_name in bum_5d:
                    cond_name = 'bum_5d'
                                
                # Get data & means 
                area = np.sum(mask, axis=(1,2)) # mask area (pixels)
                ndiff = np.sum(diff, axis=(1,2))/area # % of changed pixels
                area_mean = np.mean(area)
                area_sd = np.std(area)
                ndiff_mean = np.mean(ndiff[1:]) # exclude first 0 value
                ndiff_sd = np.std(ndiff[1:])
                
                # Append data 
                all_data.append((
                    cond_name,
                    exp_path.stem, # name of the experiment
                    idx, # cell_id
                    area, ndiff # data
                    ))
                
                # Append means
                all_means.append((
                    cond_name,
                    exp_path.stem, # name of the experiment
                    idx, # cell_id
                    area_mean, area_sd, 
                    ndiff_mean, ndiff_sd  
                    ))                

#%% Format data (DataFrame)

all_data = pd.DataFrame(all_data)
all_data.columns = [
    'cond_name',
    'exp_name', 
    'cell_id', 
    'area', 
    'change_ratio',
    ]

all_means = pd.DataFrame(all_means)
all_means.columns = [
    'cond_name',
    'exp_name', 
    'cell_id', 
    'area_mean',
    'area_sd',
    'change_ratio_mean',
    'change_ratio_sd'
    ]

#%% Get exp_means

exp_means = []

exp_names = all_means.exp_name.unique()

for exp_name in exp_names:
    
    # get exp. and cond. names
    if exp_name in ctrl_1d:
        cond_name = 'ctrl_1d' 
    if exp_name in ctrl_5d:
        cond_name = 'ctrl_5d'
    if exp_name in bum_1d:
        cond_name = 'bum_1d'
    if exp_name in bum_5d:
        cond_name = 'bum_5d'
    
    temp = all_means[all_means['exp_name'] == exp_name]
    
    exp_means.append((
        cond_name,
        exp_name,
        np.mean(temp['area_mean']),
        np.mean(temp['area_sd']),
        np.mean(temp['change_ratio_mean']),
        np.mean(temp['change_ratio_sd']),
        ))

exp_means = pd.DataFrame(exp_means)
exp_means.columns = [
    'cond_name',
    'exp_name', 
    'area_mean',
    'area_sd',
    'change_ratio_mean',
    'change_ratio_sd'
    ]    

#%% Plot & stat results (all_means)

labels = ['ctrl', 'bum']

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 9), dpi=100)
ax1.set_title('1 day post injury')
ax1.set_ylim(0,1)
ax1.boxplot([
    all_means[all_means['cond_name'] == 'ctrl_1d']['change_ratio_mean'],
    all_means[all_means['cond_name'] == 'bum_1d']['change_ratio_mean'],
    ], widths=0.75, labels=labels)

ax2.set_title('5 days post injury')
ax2.set_ylim(0,1)
ax2.boxplot([
    all_means[all_means['cond_name'] == 'ctrl_5d']['change_ratio_mean'],
    all_means[all_means['cond_name'] == 'bum_5d']['change_ratio_mean'],
    ], widths=0.75, labels=labels)

# Stats
stat_1d = stats.ttest_ind(
    all_means[all_means['cond_name'] == 'ctrl_1d']['change_ratio_mean'],
    all_means[all_means['cond_name'] == 'bum_1d']['change_ratio_mean'],
    )
print(f'all_means : ctrl_1d = bum_1d ; p-value = {stat_1d[1]:.3e}')

stat_5d = stats.ttest_ind(
    all_means[all_means['cond_name'] == 'ctrl_5d']['change_ratio_mean'],
    all_means[all_means['cond_name'] == 'bum_5d']['change_ratio_mean'],
    )
print(f'all_means : ctrl_5d = bum_5d ; p-value = {stat_5d[1]:.3e}')

stat_ctrl = stats.ttest_ind(
    all_means[all_means['cond_name'] == 'ctrl_1d']['change_ratio_mean'],
    all_means[all_means['cond_name'] == 'ctrl_5d']['change_ratio_mean'],
    )
print(f'all_means : ctrl_1d = ctrl_5d ; p-value = {stat_ctrl[1]:.3e}')

stat_bum = stats.ttest_ind(
    all_means[all_means['cond_name'] == 'bum_1d']['change_ratio_mean'],
    all_means[all_means['cond_name'] == 'bum_5d']['change_ratio_mean'],
    )
print(f'all_means : bum_1d = bum_5d ; p-value = {stat_bum[1]:.3e}')

all_means_stat = [
    ['ctrl_1d = bum_1d', stat_1d[1]],
    ['ctrl_5d = bum_5d', stat_5d[1]],
    ['ctrl_1d = ctrl_5d', stat_ctrl[1]],
    ['bum_1d = bum_5d', stat_bum[1]],
    ] 

all_means_stat = pd.DataFrame(
    all_means_stat,
    columns=['hypothesis', 'p-value']
    )


#%% Plot & stat results (exp_means)

labels = ['ctrl', 'bum']

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 9), dpi=100)
ax1.set_title('1 day post injury')
ax1.set_ylim(0,1)
ax1.boxplot([
    exp_means[exp_means['cond_name'] == 'ctrl_1d']['change_ratio_mean'],
    exp_means[exp_means['cond_name'] == 'bum_1d']['change_ratio_mean'],
    ], widths=0.75, labels=labels)

ax2.set_title('5 days post injury')
ax2.set_ylim(0,1)
ax2.boxplot([
    exp_means[exp_means['cond_name'] == 'ctrl_5d']['change_ratio_mean'],
    exp_means[exp_means['cond_name'] == 'bum_5d']['change_ratio_mean'],
    ], widths=0.75, labels=labels)

# Stats
stat_1d = stats.ttest_ind(
    exp_means[exp_means['cond_name'] == 'ctrl_1d']['change_ratio_mean'],
    exp_means[exp_means['cond_name'] == 'bum_1d']['change_ratio_mean'],
    )
print(f'exp_means : ctrl_1d = bum_1d ; p-value = {stat_1d[1]:.3e}')

stat_5d = stats.ttest_ind(
    exp_means[exp_means['cond_name'] == 'ctrl_5d']['change_ratio_mean'],
    exp_means[exp_means['cond_name'] == 'bum_5d']['change_ratio_mean'],
    )
print(f'exp_means : ctrl_5d = bum_5d ; p-value = {stat_5d[1]:.3e}')

stat_ctrl = stats.ttest_ind(
    exp_means[exp_means['cond_name'] == 'ctrl_1d']['change_ratio_mean'],
    exp_means[exp_means['cond_name'] == 'ctrl_5d']['change_ratio_mean'],
    )
print(f'exp_means : ctrl_1d = ctrl_5d ; p-value = {stat_ctrl[1]:.3e}')

stat_bum = stats.ttest_ind(
    exp_means[exp_means['cond_name'] == 'bum_1d']['change_ratio_mean'],
    exp_means[exp_means['cond_name'] == 'bum_5d']['change_ratio_mean'],
    )
print(f'exp_means : bum_1d = bum_5d ; p-value = {stat_bum[1]:.3e}')

exp_means_stat = [
    ['ctrl_1d = bum_1d', stat_1d[1]],
    ['ctrl_5d = bum_5d', stat_5d[1]],
    ['ctrl_1d = ctrl_5d', stat_ctrl[1]],
    ['bum_1d = bum_5d', stat_bum[1]],
    ] 

exp_means_stat = pd.DataFrame(
    exp_means_stat,
    columns=['hypothesis', 'p-value']
    )

#%% Save DataFrame as csv

all_means.to_csv('data/all_means.csv')
exp_means.to_csv('data/exp_means.csv')
all_means_stat.to_csv('data/all_means_stat.csv')
exp_means_stat.to_csv('data/exp_means_stat.csv')