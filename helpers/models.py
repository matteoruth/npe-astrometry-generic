import torch
import torch.nn as nn
from torch import Tensor

from lampe.nn import ResMLP
from lampe.inference import NPE

class NPEWithResMLP(nn.Module): 
    def __init__(self, 
                 num_obs, 
                 embedding_output_dim,
                 embedding_hidden_features,
                 activation,
                 transforms,
                 flow,
                 NPE_hidden_features,
                 mass_as_input = False):
        
        super().__init__()
        
        if mass_as_input: 
            num_obs = num_obs * 2 + 1
            parameters = 7
        else:
            num_obs = num_obs * 2
            parameters = 8
                   
        self.embedding = nn.Sequential(ResMLP(num_obs,
                                              embedding_output_dim,
                                              hidden_features = 3*[embedding_hidden_features],
                                              activation = activation))
        
        
        self.npe = NPE(parameters,
                       embedding_output_dim, 
                       transforms = transforms, 
                       build = flow, 
                       hidden_features = NPE_hidden_features, 
                       activation = activation)
        
    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.npe(theta, self.embedding(x))

    def flow(self, x: Tensor): 
        # The tanh function is used to ensure that the output stays between -1 and 1
        return nn.Tanh(self.npe.flow(self.embedding(x)))
    
class NPEWithDeepSet(nn.Module):
    def __init__(self, 
                 num_obs, 
                 embedding_output_dim,
                 embedding_hidden_features,
                 activation,
                 transforms,
                 flow,
                 NPE_hidden_features,
                 mass_as_input = False):
        
        super().__init__()

        if mass_as_input: 
            input = 19
            parameters = 7
        else:
            input = 18
            parameters = 8

        self.feature_extractor = nn.Sequential(
            nn.Linear(input, embedding_hidden_features),
            activation,
            nn.Linear(embedding_hidden_features, embedding_hidden_features),
            activation,
            nn.Linear(embedding_hidden_features, embedding_hidden_features)
        )

        self.regressor = nn.Sequential(
            nn.Linear(embedding_hidden_features, embedding_hidden_features),
            activation,
            nn.Linear(embedding_hidden_features, embedding_hidden_features),
            activation,
            nn.Linear(embedding_hidden_features, embedding_hidden_features),
            activation,
            nn.Linear(embedding_hidden_features, embedding_output_dim),
        ) 

        self.npe = NPE(parameters, # The 7 parameters of an orbit
                       embedding_output_dim, 
                       transforms = transforms, 
                       build = flow, 
                       hidden_features = NPE_hidden_features, 
                       activation = activation)

    def forward(self, theta: Tensor, inputs: Tensor) -> Tensor:
        inputs = inputs.to(torch.float32)

        # If the first element of the input is 0, it means it should be masked
        mask = (inputs[:,:,0] != 0)

        x = self.feature_extractor(inputs)
        x *= mask.unsqueeze(-1)


        x = x.sum(dim=1)
        x = self.regressor(x)

        return self.npe(theta, x)
    
    def flow(self, inputs: Tensor):
        
        inputs = inputs.to(torch.float32)

        # If the first element of the input is 0, it means it should be masked
        mask = (inputs[:,:,0] != 0)

        x = self.feature_extractor(inputs)
        x *= mask.unsqueeze(-1)


        x = x.sum(dim=1)
        x = self.regressor(x)
        
        # The tanh function is used to ensure that the output stays between -1 and 1
        return nn.Tanh(self.npe.flow(x))