from helpers import Prior, train 
from lampe.data import H5Dataset
from orbitize import DATADIR, read_input
import torch.nn as nn
import zuko 
import argparse
import wandb

def main(method):

    trainset = H5Dataset(f'datasets/generic-{method}-train.h5', batch_size=2048, shuffle=True)  
    validset = H5Dataset(f'datasets/generic-{method}-val.h5', batch_size=2048, shuffle=True)
    
    LOWER = [4.0, 1e-5, 180.0, 0.0, 0.0, 0.0, 50.0, 0.5]
    UPPER = [200.0, 0.99, 0.0, 360.0, 360.0, 1.0, 55.0, 3.0]

    priors = Prior(LOWER, UPPER) # Needed to post-process the data in the training phase

    num_obs = 180 # changer ça peut être pour que ce ne soit pas hardcodé
    
    wandb.login()
    train(
        trainset=trainset,
        validset=validset,
        prior=priors,
        epochs=1024,
        NPE_hidden_features=[512] * 5,
        flow = zuko.flows.spline.NSF,
        transforms = 3,
        num_obs=num_obs,
        mass_as_input=False,
        use_wandb=True,
        embedding=method
    )   

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--method", type=str, default="ResMLP", help="The method to use for the simulator")
    args = parser.parse_args()

    main(args.method)
