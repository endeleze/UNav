import numpy as np
import torch
import cv2

from core.third_party.SuperPoint_SuperGlue.base_model import dynamic_load
from core.third_party.SuperPoint_SuperGlue import extractors, matchers
from core.third_party.LightGlue.lightglue import LightGlue

class Superpoint:
    """
    SuperPoint local feature extractor wrapper.
    """
    def __init__(self, device, conf):
        """
        Args:
            device (str): Device string ("cuda" or "cpu").
            conf (dict): SuperPoint configuration dictionary.
        """
        Model_sp = dynamic_load(extractors, conf["detector_name"])
        self.local_feature_extractor = (
            Model_sp({
                "name": conf["detector_name"],
                "nms_radius": conf["nms_radius"],
                "max_keypoints": conf["max_keypoints"],
            })
            .eval()
            .to(device)
        )
        self.device = device

    def prepare_data(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert BGR image to normalized torch tensor (1, 1, H, W) for inference.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image = image[None]
        data = torch.from_numpy(image / 255.0).unsqueeze(0)
        return data

    def extract_local_features(self, image0: np.ndarray) -> dict:
        """
        Extract local features from the given image.

        Args:
            image0 (np.ndarray): Input BGR image.

        Returns:
            dict: { 'keypoints', 'scores', 'descriptors', 'image_size', ... }
        """
        data0 = self.prepare_data(image0)
        pred0 = self.local_feature_extractor(data0.to(self.device))
        del data0
        torch.cuda.empty_cache()
        pred0 = {k: v[0].cpu().detach().numpy() for k, v in pred0.items()}
        if "keypoints" in pred0:
            pred0["keypoints"] = (pred0["keypoints"] + 0.5) - 0.5
        pred0["image_size"] = np.array([image0.shape[1], image0.shape[0]])
        return pred0

class Local_extractor:
    """
    Factory for local feature extractors and matchers.
    Supports SuperPoint+SuperGlue, SuperPoint+LightGlue, and extension to SIFT/SURF.
    """
    def __init__(self, configs: dict):
        """
        Args:
            configs (dict): Configuration for all extractors/matchers.
        """
        self.configs = configs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def lightglue(self, conf: dict):
        """
        Initialize LightGlue matcher.

        Args:
            conf (dict): LightGlue configuration.
        Returns:
            LightGlue model (callable)
        """
        return LightGlue(pretrained="superpoint", **conf["match_conf"]).eval()

    def superglue(self, conf: dict):
        """
        Initialize SuperGlue matcher.

        Args:
            conf (dict): SuperGlue configuration.
        Returns:
            SuperGlue model (callable)
        """
        Model_sg = dynamic_load(matchers, conf["matcher_name"])
        return Model_sg({
            "name": conf["matcher_name"],
            "weights": conf["weights"],
            "sinkhorn_iterations": conf["sinkhorn_iterations"],
        }).eval()

    def extractor(self):
        """
        Returns the local feature extractor function for the specified configuration.

        Returns:
            Callable: Function that takes an image and returns extracted features.
        """
        for name, content in self.configs.items():
            if name == "superpoint+superglue":
                superpoint = Superpoint(self.device, self.configs["superpoint+superglue"])
                return superpoint.extract_local_features
            elif name == "superpoint+lightglue":
                superpoint = Superpoint(self.device, self.configs["superpoint+lightglue"])
                return superpoint.extract_local_features
            elif name == "sift":
                # TODO: Implement SIFT extractor if needed
                pass
            elif name == "surf":
                # TODO: Implement SURF extractor if needed
                pass
        raise ValueError("No supported local extractor config found.")

    def matcher(self):
        """
        Returns the local feature matcher for the specified configuration.

        Returns:
            Callable: Matcher model.
        """
        for name, content in self.configs.items():
            if name == "superpoint+superglue":
                return self.superglue(self.configs["superpoint+superglue"])
            elif name == "superpoint+lightglue":
                return self.lightglue(self.configs["superpoint+lightglue"])
            elif name == "sift":
                # TODO: Implement SIFT matcher if needed
                pass
            elif name == "surf":
                # TODO: Implement SURF matcher if needed
                pass
        raise ValueError("No supported matcher config found.")
