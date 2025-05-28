import os
from os.path import join

from unav.core.third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
from unav.core.third_party.MixVPR.feature_extract import MixVPRFeatureExtractor
from unav.core.third_party.AnyLoc.feature_extract import VLADDinoV2FeatureExtractor
from unav.core.third_party.salad.feature_extract import DINOV2SaladFeatureExtractor
from unav.core.third_party.CricaVPR.feature_extract import CricaVPRFeatureExtractor

class GlobalExtractors:
    """
    Factory for instantiating and managing multiple global descriptor models.

    Usage:
        g_extractor = GlobalExtractors(root, {'MixVPR': {...}})
        features = g_extractor('MixVPR', image_tensor)
    """
    model_classes = {
        "MixVPR": MixVPRFeatureExtractor,
        "AnyLoc": VLADDinoV2FeatureExtractor,
        "DinoV2Salad": DINOV2SaladFeatureExtractor,
        "CricaVPR": CricaVPRFeatureExtractor
    }

    def __init__(self, root, g_extr_conf, pipeline=False, data_parallel=False):
        """
        Initialize all requested models using their configs.

        Args:
            root (str): Path prefix to model weights.
            g_extr_conf (dict): E.g., {'MixVPR': {...}}, {'NetVlad': {...}}
            pipeline (bool): If True, set models to pipeline mode if supported.
            data_parallel (bool): If True, wrap models for DataParallel execution.
        """
        self.models_objs = {}
        for model_name, model_conf in g_extr_conf.items():
            if model_name == "NetVlad":
                netvlad_model = NetVladFeatureExtractor(
                    os.path.join(root, model_conf["ckpt_path"]),
                    type="pipeline" if pipeline else None,
                    arch=model_conf['arch'],
                    num_clusters=model_conf['num_clusters'],
                    pooling=model_conf['pooling'],
                    vladv2=model_conf['vladv2'],
                    nocuda=model_conf['nocuda']
                )
                if data_parallel:
                    netvlad_model.set_parallel()
                self.models_objs["NetVlad"] = netvlad_model
            elif model_name in GlobalExtractors.model_classes:
                model_class = GlobalExtractors.model_classes[model_name]
                model = model_class(root, model_conf, pipeline)
                if data_parallel:
                    model.set_parallel()
                self.models_objs[model_name] = model
            else:
                print(f"[Warning] {model_name} not implemented, skipped.")

    def __call__(self, request_model, images):
        """
        Apply the requested model to the given images.

        Args:
            request_model (str): Name of the initialized model.
            images (Tensor/np.ndarray): Input images.

        Returns:
            Model output (features).
        """
        if request_model not in self.models_objs:
            raise ValueError(
                f"Model '{request_model}' not initialized! Available: {list(self.models_objs.keys())}"
            )
        return self.models_objs[request_model](images)

    def set_train(self, is_train: bool):
        """
        Set train/eval mode for all models.

        Args:
            is_train (bool): True for train, False for eval.
        """
        for model in self.models_objs.values():
            model.set_train(is_train)

    def torch_compile(self, **compile_args):
        """
        Apply torch.compile to all models if supported.

        Args:
            **compile_args: Compilation options.
        """
        for model in self.models_objs.values():
            model.torch_compile(**compile_args)

    def set_float32(self):
        """
        Convert all models to float32 precision if supported.
        """
        for model in self.models_objs.values():
            model.set_float32()

    def save_state(self, model, save_path, new_state):
        """
        Save model state.

        Args:
            model (str): Model name.
            save_path (str): Path to save.
            new_state: State dict.
        """
        if model not in self.models_objs:
            raise ValueError(f"Model '{model}' not initialized!")
        self.models_objs[model].save_state(save_path, new_state)

    @property
    def models(self):
        """
        Returns list of initialized model names.
        """
        return list(self.models_objs.keys())

    def last_epoch(self, model):
        return self.models_objs[model].last_epoch

    def best_score(self, model):
        return self.models_objs[model].best_score

    def model_parameters(self, model):
        return self.models_objs[model].parameters

    def feature_length(self, model):
        return self.models_objs[model].feature_length
