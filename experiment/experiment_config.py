import os
import json

def import_config(fn) -> dict:
    d = None
    with open(fn, 'r') as f:
        d = json.load(f)
    return d

def export_config(name, config_dict, fn, output_path, WRITE=False):
    if WRITE:
        os.makedirs(output_path, exist_ok=True)
        with open(output_path+'/'+fn+'.json', 'w') as file:
             file.write(json.dumps(config_dict))
    else:
        print(f'({name})\t{output_path}/{fn}')
        print(*config_dict.items(), sep='\n')
    return output_path+'/'+fn

def main():
    # datasets = [
    #     {'dataset_name':'TestSet_360_v1.0_a',
    #      'FLOOR':'1.0_MixVPR',
    #      'GROUND_TRUTH_IMG_LIST':'/home/nattachart.tak/Data/experiments/groundtruth_img_dataset_ICT_Sample_1.0a_10P.csv',
    #      'path_to_images':f'/home/nattachart.tak/Data/experiments/Mapping/data/src_images/Mahidol_University/ICT/1.0_MixVPR/perspective_images',
    #     },
    #     {'dataset_name':'TestSet_360_v1.0_b',
    #      'FLOOR':'1.0_MixVPR',
    #      'GROUND_TRUTH_IMG_LIST':'/home/nattachart.tak/Data/experiments/groundtruth_img_dataset_ICT_Sample_1.0b_10P.csv',
    #      'path_to_images':f'/home/nattachart.tak/Data/experiments/Mapping/data/src_images/Mahidol_University/ICT/1.0_MixVPR/perspective_images',
    #     },
    #     {'dataset_name':'TestSet_360_v1.1_a',
    #      'FLOOR':'1.1_MixVPR',
    #      'GROUND_TRUTH_IMG_LIST':'/home/nattachart.tak/Data/experiments/groundtruth_img_dataset_ICT_Sample_1.1_10P.csv',
    #      'path_to_images':f'/home/nattachart.tak/Data/experiments/Mapping/data/src_images/Mahidol_University/ICT/1.0_MixVPR/perspective_images',
    #     },
    # ]
    
    
    # UNav-V2 Dataset Data:
    DATA_FINAL_ROOT = '/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data'
    GLOBAL_DESCRIPTOR_MODEL = 'NetVlad'
    LOCAL_DESCRIPTOR_MODEL = 'superpoint+lightglue'
    PLACES = {
            "Mahidol_University": {
                "ICT": ["1.1_NetVlad"]
            }
    }

    # Experriment Config Data:
    EXP_VERSION = 'v3-1_a'
    OUT_PATH = f'Config__{EXP_VERSION}'
    ROOT_DIR = f'/home/nattachart.tak/PhD/Trial_New_UNav/UNav/experiment/configs/'
    EXP_TYPE = 'localize'
    
    config = {
        'data_final_root':DATA_FINAL_ROOT,
        'global_descriptor_model':GLOBAL_DESCRIPTOR_MODEL,
        'local_feature_model':LOCAL_DESCRIPTOR_MODEL,
        'places':PLACES,
        'path_to_images':f'/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data/Mahidol_University/ICT/1.1_NetVlad/perspectives',
        'ground_truth_img_list':f'/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data/Mahidol_University/ICT/1.1_NetVlad/groundtruth_img_dataset_ICT_1.1_NetVlad_10P.csv',
        'img_list_attr':'image_name',
        'img_ext':'png',
        'exp_type':EXP_TYPE,
        'results_out_path':OUT_PATH,
        'root_dir':ROOT_DIR,
        'results_fn':f"Results_{EXP_TYPE}.xlsx",
        "num_trials": 5,
        'exp_version':EXP_VERSION,
        'exp_description':f'''
            v3-1_a - Mapping on 1.1_NetVlad and localization on the 1.1 dataset (correspondences.json copied from MixVPR's)
            v2-1_c - Mapping on 1.1 and localization on the 1.1 dataset
            v2-1_b - Change to UNav-V2 - Image localization experiment - 5 VPR algorithms. Vision only.
            v2-1_a - Baseline UNav_MixVPR - Image localization measurement. Measure time to localize (hloc.get_location(..))
        '''
    }

    export_config(EXP_VERSION, config, fn="config", output_path=f'{ROOT_DIR}{OUT_PATH}', WRITE=True)


if __name__ == '__main__':    
    main()
