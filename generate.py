from helpers import Simulator, Prior
import argparse
from lampe.data import H5Dataset, JointLoader

def main(method, size, name):
    LOWER = [4.0, 1e-5, 180.0, 0.0, 0.0, 0.0, 50.0, 0.5]
    UPPER = [200.0, 0.99, 0.0, 360.0, 360.0, 1.0, 55.0, 3.0]

    prior = Prior(LOWER, UPPER)
    simulator = Simulator(method, scale=1e6, discretisation=180, use_plx=False, use_mtot=False, prior=prior)

    loader = JointLoader(prior, simulator, batch_size=16, vectorized=True)

    H5Dataset.store(
        loader, 
        f'datasets/{name}-{method}-test.h5', 
        size=(2**size)//64, # 130.000
        overwrite=True)
    H5Dataset.store(
        loader, 
        f'datasets/{name}-{method}-val.h5', 
        size=(2**size)//8, # 1.000.000
        overwrite=True)
    H5Dataset.store(
        loader, 
        f'datasets/{name}-{method}-train.h5', 
        size=2**size, # 8.400.000
        overwrite=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate betapic datasets")
    parser.add_argument("--method", type=str, default="Deepset", help="The method to use for the simulator")
    parser.add_argument("--size", type=int, default=23, help="The exponent of 2 for the training dataset size")
    parser.add_argument("--name", type=str, default="generic", help="Base name for datasets")
    args = parser.parse_args()

    main(args.method, args.size, args.name)