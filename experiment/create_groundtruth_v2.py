import sys, os
import pandas as pd
import numpy as np
from pathlib import Path

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from unav.localizer.tools.pnp import transform_pose_to_floorplan

def parse_args() -> tuple:
    """
    Parse command-line arguments.

    Returns:
        Tuple containing:
            data_final_root (str)
            place (str)
            building (str)
            floor (str)
    """
    if len(sys.argv) != 5:
        print(
            f"Usage: python {sys.argv[0]} "
            "<data_final_root> <place> <building> <floor>"
        )
        sys.exit(1)
    return tuple(sys.argv[1:])

# ------------------- Groundtruth Section -------------------
def main():
    # ------------------- Configuration Section -------------------
    (
        data_final_root,
        place,
        building,
        floor
    ) = parse_args()

    print(f"==== Calculate the groundtruth ====")
    groundtruth_csv, groundtruth10P_csv = get_and_save_groundtruth(data_final_root, place, building, floor)
    print(f"==== The groundtruth was saved to {groundtruth_csv} and {groundtruth10P_csv} ====")

def get_and_save_groundtruth(ROOT_DIR, PLACE, BLD, FLOOR):
    #ROOT_DIR = '/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data'
    #PLACE = 'Mahidol_University'
    #BLD = 'ICT'
    #FLOOR = '1_MixVPR'
    transform_matrix = np.load(f'{ROOT_DIR}/{PLACE}/{BLD}/{FLOOR}/transform_matrix.npy')
    image_txt_cols = ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME']
    df = pd.read_csv(f'{ROOT_DIR}/{PLACE}/{BLD}/{FLOOR}/colmap_sfm/sparse/0/images.txt', 
                sep=' ', 
                header=None,
                names=image_txt_cols,
                skiprows=2)

    qvec = ['QW', 'QX', 'QY', 'QZ']
    tvec = ['TX', 'TY', 'TZ']
    r = []
    for tup in df.iterrows():
        row = tup[1]
        q = row[qvec].tolist() # 0.851773 0.0165051 0.503764 -0.142941
        t = row[tvec].tolist() # -0.737434 1.02973 3.74354
        d = transform_pose_to_floorplan(q, t, transform_matrix)
        cx, cy = d['xy'][0], d['xy'][1]
        ang = d['ang']
        image_name = Path(row['NAME']).stem
        r += [{
            'image_name': image_name, # P1180141.JPG
            'cx': cx,
            'cy': cy,
            'ang': ang,
        }]
        
    dfr = pd.DataFrame.from_records(r)
    groundtruth_csv = f'{ROOT_DIR}/{PLACE}/{BLD}/{FLOOR}/groundtruth_img_dataset_{BLD}_{FLOOR}.csv'
    groundtruth10P_csv = f'{ROOT_DIR}/{PLACE}/{BLD}/{FLOOR}/groundtruth_img_dataset_{BLD}_{FLOOR}_10P.csv'
    dfr.to_csv(groundtruth_csv)
    dfr.sample(frac=0.1).to_csv(groundtruth10P_csv)
    return groundtruth_csv, groundtruth10P_csv

if __name__ == "__main__":
    main()