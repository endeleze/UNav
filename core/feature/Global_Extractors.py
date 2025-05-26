from os.path import join
import os
from core.third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
from core.third_party.MixVPR.feature_extract import MixVPRFeatureExtractor
from core.third_party.AnyLoc.feature_extract import VLADDinoV2FeatureExtractor
from core.third_party.salad.feature_extract import DINOV2SaladFeatureExtractor
from core.third_party.CricaVPR.feature_extract import CricaVPRFeatureExtractor


class GlobalExtractors:
    model_classes = {
        "MixVPR": MixVPRFeatureExtractor,
        "AnyLoc": VLADDinoV2FeatureExtractor,
        "DinoV2Salad": DINOV2SaladFeatureExtractor,
        "CricaVPR": CricaVPRFeatureExtractor
    }

    def __init__(self, root, g_extr_conf, pipeline=False, data_parallel=False):
        """
        Initialize ONE model only from given configuration.
        - root: model parameters root directory
        - g_extr_conf: dict, e.g., {'MixVPR': {...}} or {'CricaVPR': {...}}
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
                self.models_objs["NetVlad"] = netvlad_model
                if data_parallel:
                    self.models_objs["NetVlad"].set_parallel()
            elif model_name in GlobalExtractors.model_classes:
                model_class = GlobalExtractors.model_classes[model_name]
                self.models_objs[model_name] = model_class(root, model_conf, pipeline)
                if data_parallel:
                    self.models_objs[model_name].set_parallel()
            else:
                print(f"[Warning] {model_name} not implemented, skipped.")

    def __call__(self, request_model, images):
        if request_model not in self.models_objs:
            raise ValueError(f"Model {request_model} not initialized! Available: {list(self.models_objs.keys())}")
        return self.models_objs[request_model](images)

    def set_train(self, is_train):
        for model in self.models_objs.values():
            model.set_train(is_train)

    def torch_compile(self, **compile_args):
        for model in self.models_objs.values():
            model.torch_compile(**compile_args)

    def set_float32(self):
        for model in self.models_objs.values():
            model.set_float32()

    def save_state(self, model, save_path, new_state):
        self.models_objs[model].save_state(save_path, new_state)

    @property
    def models(self):
        return list(self.models_objs.keys())

    def last_epoch(self, model):
        return self.models_objs[model].last_epoch

    def best_score(self, model):
        return self.models_objs[model].best_score

    def model_parameters(self, model):
        return self.models_objs[model].parameters

    def feature_length(self, model):
        return self.models_objs[model].feature_length
