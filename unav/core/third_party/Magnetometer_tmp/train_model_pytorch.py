import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MagModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(MagModel, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 20)
        self.fc2 = nn.Linear(20, 5)
        self.fc3 = nn.Linear(5, n_outputs)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')  # he_uniform
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='linear')

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


if __name__ == '__main__':
        
    # sys.path.append("../Magnetometer")
    # sys.path.append('../Magnetometer/processing_blocks_master/')
    # sys.path.append('../Magnetometer/processing_blocks_master/spectral_analysis/')
    # from preprocess import get_spectral_analysis_v5_features
    num_inputs = 39
    
    batch_sample = np.random.sample(150).reshape(50,3)
    model = MagModel( n_inputs=num_inputs, n_outputs=2 )
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    torch.save(model.state_dict(), "unfitted_pytorch.pth")
