import sys
import torch
import numpy as np

from preprocess import get_spectral_analysis_v5_features
from train_model_pytorch import MagModel
import models

class MagFeatureExtractor:
    def __init__(self, root, content, pipeline=False):
        self.device = "cuda" if content["cuda"] else "cpu"
        self.load_model(root, content, pipeline)
        self.config = content
        
    
    def load_model(self, root, content, pipeline):
        self.model = models.MagneticFieldModel_1(n_inputs=content['n_inputs'], n_outputs=content['n_outputs'])
        self.model.load_truncated_state_dict(torch.load(content['model_path']))
        self.model.eval().to(self.device)

    def preprocess(self, batch:list) -> list:
        features = None
        if self.config['preprocess'] == 'SpectralV5':
            features = get_spectral_analysis_v5_features( batch )
        else:
            raise NotImplementedError('Select a valid preprocessing method.')
        return features
        
    def __call__(self, batch:np.array):
        tensor_sample = torch.tensor( self.preprocess(batch), dtype=torch.float32 ).to(self.device)
        pred = self.model( tensor_sample )
        return pred, None

    # def set_train(self, is_train):
    #     self.model.train(is_train)
    
    # def torch_compile(self, **compile_args):
    #     self.model = torch.compile(self.model, **compile_args)
    
    # def set_parallel(self):
    #     self.model = torch.nn.DataParallel(self.model)
    
    # def set_float32(self):
    #     self.model.to(torch.float32)
    
    # def save_state(self, save_path, new_state):
    #     new_state["state_dict"] = self.model.state_dict()
    #     torch.save(new_state, save_path)
    
    # @property
    # def last_epoch(self): return self.saved_state["epoch"]
    
    # @property
    # def best_score(self): return self.saved_state["best_score"]
    
    # @property
    # def parameters(self): return self.model.parameters()
    
    # @property
    # def feature_length(self):
    #     return 14 * 768    # patch_size * embed_dim



if __name__ == "__main__":
    root = "/home/nattachart.tak/Data/experiments/Mapping/data/unav2-data/"
    content = {
            "model_path": '/home/nattachart.tak/PhD/Trial_New_UNav/UNav/unav/core/third_party/Magnetometer/unfitted_pytorch.pth',
            'n_outputs':2,
            'n_inputs':39
            }

    import numpy as np
    import torch
    m = MagFeatureExtractor(root, content)
    batch_sample = np.random.sample(150).reshape(50,3)
    print('Input Batch:', batch_sample.shape)
    print(batch_sample)
    tensor_sample = tensor_img = torch.from_numpy(batch_sample)
    features = m(tensor_sample)
    print('Features:')
    print(features)