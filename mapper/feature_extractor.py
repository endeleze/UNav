import os
import h5py
import cv2
import torch
from tqdm import tqdm
from torch.nn.functional import normalize
from core.feature.Global_Extractors import GlobalExtractors
from core.feature.local_extractor import Local_extractor
from config import UNavMappingConfig
from typing import Dict


def extract_features_from_dir(config: UNavMappingConfig) -> None:
    """
    Extract local & global features directly from input_perspective_dir.
    Config must follow UNavMappingConfig.feature_extraction_config format,
    which must provide all paths including:
    - input_perspective_dir
    - local_feat_save_path
    - global_feat_save_path
    """
    feat_cfg = config.feature_extraction_config
    
    # Ensure output dir exists (already defined in feat_cfg)
    os.makedirs(feat_cfg["output_feature_dir"], exist_ok=True)

    # Prepare image list
    img_list = sorted([f for f in os.listdir(feat_cfg["input_perspective_dir"]) if f.endswith('.png')])

    # Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print(f"[INFO] Loading models: Local -> {feat_cfg['local_feature_model']} | Global -> {feat_cfg['global_descriptor_model']}")
    local_extractor = Local_extractor(feat_cfg["local_extractor_config"]).extractor()
    global_extractor = GlobalExtractors(
        feat_cfg['parameters_root'],
        {feat_cfg["global_descriptor_model"]: feat_cfg["global_descriptor_config"]},
        data_parallel=False
    )
    global_extractor.set_train(False)

    # Extraction loop
    with h5py.File(feat_cfg["local_feat_save_path"], "a") as local_h5, \
         h5py.File(feat_cfg["global_feat_save_path"], "a") as global_h5:

        for img_name in tqdm(img_list, desc="Extracting Features"):
            img_path = os.path.join(feat_cfg["input_perspective_dir"], img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warning] Cannot read {img_path}, skip.")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Global Descriptor
            if img_name not in global_h5:
                tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                feat = global_extractor(feat_cfg["global_descriptor_model"], tensor_img)
                if isinstance(feat, tuple):
                    feat = feat[1]
                feat = normalize(feat, dim=-1).squeeze(0).detach().cpu().numpy()
                global_h5.create_dataset(img_name, data=feat, compression="gzip")

            # Local Feature
            if img_name not in local_h5:
                feat_dict = local_extractor(img_rgb)
                feat_grp = local_h5.create_group(img_name)
                for k, v in feat_dict.items():
                    feat_grp.create_dataset(k, data=v, compression="gzip")

    print(f"✅ Features saved: Local -> {feat_cfg['local_feat_save_path']} | Global -> {feat_cfg['global_feat_save_path']}")
