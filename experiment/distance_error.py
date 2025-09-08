import pandas as pd
import cv2
import os
import json
import numpy as np
import sys
from scipy.spatial import distance
from os.path import dirname,join,exists,realpath


def save_calculated_distance_error(exp_conf, prediction_filename):
    pixel_to_meter_ratio = 27.14
    dataset_name = 'TestSet_360_v1.1_a'
    algorithm = exp_conf['global_descriptor_model']
    root_dir = exp_conf['root_dir']
    
    est_df = pd.read_excel(os.path.join(root_dir, exp_conf['results_out_path'], prediction_filename))
    est_df['image_name'] = est_df['image_fn'].apply(lambda s : s.split('/')[-1].split('.')[0])
    est_df = est_df.drop(columns=['error'],axis=1)
    ground_truth = pd.read_csv(exp_conf['ground_truth_img_list'])
    
    eu_dist_error_df = est_df.set_index('image_name').dropna().join(ground_truth.set_index('image_name'), how='inner', lsuffix='_est', rsuffix='_ground').apply(
        lambda row : distance.euclidean([row['coordx'], row['coordy']], [row['cx'], row['cy']]), axis=1).to_frame()

    eu_dist_error_df['time_ms'] = est_df.set_index('image_name').dropna()['time_ms']
    eu_dist_error_df['trial_num'] = est_df.set_index('image_name').dropna()['trial_num']
    eu_dist_error_df = eu_dist_error_df.rename(columns={0:'error_distance_pixel'})
    eu_dist_error_df['error_distance_meter'] = eu_dist_error_df['error_distance_pixel'] / pixel_to_meter_ratio
    eu_dist_error_df = eu_dist_error_df[['error_distance_pixel','error_distance_meter','time_ms','trial_num']]
    
    out_path = exp_conf['results_out_path']
    distance_error_fn = f"{root_dir}/{out_path}/Results_distance_error_{algorithm}.xlsx"
    eu_dist_error_df.to_excel(distance_error_fn) # Save errors to Excel
    print(f"==== Distance errors were saved to {distance_error_fn} ====")
    means = eu_dist_error_df[['error_distance_meter','trial_num']].groupby('trial_num').mean()['error_distance_meter'].tolist()
    stds = eu_dist_error_df[['error_distance_meter','trial_num']].groupby('trial_num').std()['error_distance_meter'].tolist()

    means_time_ms = (eu_dist_error_df[['time_ms','trial_num']].groupby('trial_num').mean()['time_ms'] / (len(est_df)/exp_conf['num_trials'])).tolist()
    stds_time_ms = (eu_dist_error_df[['time_ms','trial_num']].groupby('trial_num').std()['time_ms'] / (len(est_df)/exp_conf['num_trials'])).tolist()

    r = {'exp_version' : exp_conf['exp_version'],      
            'distance_error_meters_mean_mean' : np.nanmean(means),
            'distance_error_meters_std_mean' : np.nanmean(stds),
            'distance_error_meters_per_trial_mean' : means,
            'distance_error_meters_per_trial_std' : stds,      
            'time_localization_mean_mean' : np.nanmean(means_time_ms),
            'time_localization_std_mean' : np.nanmean(stds_time_ms),
            'time_localization_mean_per_trial' : means_time_ms,
            'time_localization_std_per_trial' : stds_time_ms,
            'num_images_localized' : len(eu_dist_error_df),
            'num_images_total' : len(est_df),
            'perc_images_localized' : (len(eu_dist_error_df)/exp_conf['num_trials']) / (len(est_df)/exp_conf['num_trials']),
            'num_images_per_trial' : len(est_df)/exp_conf['num_trials'],
            'num_trials': exp_conf['num_trials'],
            'dataset_name' : dataset_name,
            'algorithm' : algorithm
        }

    with open(f"{root_dir}/{out_path}/distance_error_meta_{algorithm}.json", 'w') as f:
        f.write(json.dumps(r))

    return distance_error_fn