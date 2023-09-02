from os.path import join
import os
import torch
import sys
from third_party.global_feature.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor

mixvpr_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'third_party', 'global_feature', 'MixVPR')
sys.path.append(mixvpr_path)
from main import VPRModel

class Global_Extractors():
    def __init__(self, root,config):
        self.root=root
        self.extractor = config

    def netvlad(self, content):
        return NetVladFeatureExtractor(join(self.root,content['ckpt_path']), arch=content['arch'],
         num_clusters=content['num_clusters'],
         pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])
    
    def mixvpr(self,content):
        model = VPRModel(backbone_arch=content["backbone_arch"], 
                        layers_to_crop=[content['layers_to_crop']],
                        agg_arch=content['agg_arch'],
                        agg_config=content['agg_config'],
                        )

        state_dict = torch.load(content["ckpt_path"])
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def vlad(self, contend):
        pass

    def bovw(self, contend):
        pass

    def get(self):
        for extractor, content in self.extractor.items():
            if extractor == 'netvlad':
                return self.netvlad(content).feature
            if extractor == 'mixvpr':
                return self.mixvpr(content)
            if extractor == 'vlad':
                pass
            if extractor == 'bovw':
                pass
