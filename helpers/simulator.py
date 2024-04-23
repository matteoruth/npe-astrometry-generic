r""" This is the simulator that computes the right ascension and
declination of an exoplanet given the orbital parameters and the epochs of
observation.

The orbit calculator from the `orbitize.kepler` module  is used to do the
computations. The simulator is a callable class that takes in a tensor of
orbital parameters and returns a tensor of right ascension and declination
values.
"""

import torch
import torch.nn.functional as F
import numpy as np
from astropy.time import Time
from orbitize.kepler import calc_orbit

def getPositionEncoding(time):
    batch_size, nbr_obs = time.shape
    
    ref_time = 52275  # 01/01/02
    n = 10000
    d = 16  # length of the token 
    
    k = time - ref_time
    i = torch.arange(int(d/2), dtype=torch.float32)
    
    denominator = torch.pow(n, 2*i/d)#.cuda()
    sin_encod = torch.sin(k.view(-1, 1)/denominator)
    cos_encod = torch.cos(k.view(-1, 1)/denominator)

    P = torch.zeros((batch_size * nbr_obs, d), dtype=torch.float32)
    P[:, 0::2] = sin_encod
    P[:, 1::2] = cos_encod
    
    P = P.view(batch_size, nbr_obs, d)
    return P

class Simulator:
    r""" 
    Creates an orbit simulator.
    """

    def __init__(self, 
                 method,
                 scale=1e6, 
                 discretisation=180,
                 use_plx = False, 
                 use_mtot = False,
                 prior=None):
        
        valid = {"ResMLP", "Deepset"}
        if method not in valid:
            raise ValueError("method: status must be one of %r." % valid)
        else :
            self.method = method

        if use_mtot and not prior:
            raise ValueError("If use_mtot is True, prior must be provided.")
        else:
            self.prior = prior

        self.epochs = Time(np.linspace(2002, 2019.9, discretisation + 1), 
                           format="decimalyear").mjd
        self.number_epochs = discretisation
        self.scale = scale
        self.use_plx = use_plx  
        self.use_mtot = use_mtot

    def __call__(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Simulates observations of astronomical objects given a set of input parameters.
        """
        n_samples, _ = thetas.shape

        theta = thetas.T
        sma, ecc, inc, aop, pan, tau, plx, mtot = theta.numpy()

        if not self.use_plx:
            plx = 100 # it will later be scaled back to the known parallax
        
        inc = np.radians(inc)
        aop = np.radians(aop)
        pan = np.radians(pan)

        # Randomly sample epochs between the discretised epochs,
        # note that it will be the same for all samples from the same batch
        epochs = np.array([np.random.uniform(self.epochs[i], self.epochs[i+1]) for i in range(self.number_epochs)])
        
        ra, dec, _ = calc_orbit(
            epochs, 
            sma, 
            ecc, 
            inc, 
            aop, 
            pan,
            tau, 
            plx, 
            mtot, 
            use_gpu = True, 
            tau_ref_epoch=50000)

        # Adding Gaussian offets from observational uncertainties
        ra_err = 2
        dec_err = 2
        ra_err = np.random.normal(0, ra_err, (n_samples, self.number_epochs))
        dec_err = np.random.normal(0, dec_err, (n_samples, self.number_epochs))

        ra = ra.T + ra_err
        dec = dec.T + dec_err

        if self.use_mtot:
            _, _, _, _, _, _, _, mtot_processed = self.prior.pre_process(thetas).T
        
        if self.method == "ResMLP":
            # calc_orbit returns numpy arrays, need to convert to tensors
            ra = torch.tensor(ra,dtype=torch.float32)
            dec = torch.tensor(dec,dtype=torch.float32)

            interleaved_tensor = torch.stack((ra, dec), dim=2)
            interleaved_tensor = interleaved_tensor.view(n_samples, -1)

            # Randomly set some observations to zero to simulate missing data
            # Keep between 3 and 30 observations, as this is a reasonable range
            # for the number of observations of an exoplanet using direct imaging
            nbr_observations = np.random.randint(low=3, high=31, size=n_samples)

            mask = torch.zeros((n_samples, 2 * self.number_epochs), dtype=torch.float32)
            
            for i, num_observation in enumerate(nbr_observations):
                indices = np.random.choice(180, num_observation, replace=False)
                mask[i, indices * 2] = 1
                mask[i, indices * 2 + 1] = 1
            
            interleaved_tensor = torch.where(mask.bool(), interleaved_tensor, torch.tensor(0, dtype=torch.float32))

            results = self.process(interleaved_tensor)
            
            if self.use_mtot:
                print(mtot_processed.unsqueeze(1).shape)
                print(results.shape)
                results = torch.cat((mtot_processed.unsqueeze(1), results), dim=1)

        elif self.method == "Deepset":
            indices = []

            num_observation = np.random.randint(low=3, high=30)
            for _ in range(n_samples):
                sample_indices = np.random.choice(self.number_epochs, replace=False, size=num_observation)
                sample_indices = np.sort(sample_indices)
                indices.append(sample_indices)
            
            ra = ra[np.arange(n_samples)[:, None], indices]
            dec = dec[np.arange(n_samples)[:, None], indices]
            epochs = epochs[indices]
            
            # calc_orbit returns numpy arrays, need to convert to tensors
            ra = torch.tensor(ra)
            dec = torch.tensor(dec)
            epochs = torch.tensor(epochs)
            
            # use transformer positional encoding for the epochs
            epochs = getPositionEncoding(epochs)

            # pad the results with zeros to have 30 observations
            num_zeros = 30 - num_observation
            ra = F.pad(ra, (0, num_zeros), value=0)
            dec = F.pad(dec, (0, num_zeros), value=0)
            epochs = F.pad(epochs, (0, 0, 0, num_zeros, 0, 0), value=0)

            if self.use_mtot:
                mtot_processed = mtot_processed.unsqueeze(1).repeat(1, num_observation)
                mtot_processed = F.pad(mtot_processed, (0, num_zeros), value=0)
                results = torch.cat([mtot_processed.unsqueeze(2), 
                                     self.process(ra).unsqueeze(2), 
                                     self.process(dec).unsqueeze(2), 
                                     epochs], dim=2)
            else:
                results = torch.cat([self.process(ra).unsqueeze(2), 
                                     self.process(dec).unsqueeze(2), 
                                     epochs], dim=2)

        return results
    
    def process(self, results: torch.Tensor) -> torch.Tensor:
        r""" Processes the results of the simulator

        Process the results of the simulator to make them more suitable for
        training. 

        Just dividing by the scale factor is enough because the simulated 
        data follows a Laplacian distribution which is symmetric at each 
        timestep. The demonstration is trivial. 

        Arguments:
            results (torch.Tensor): Tensor of right ascension and declination
                values generated by the simulator. This tensor is of shape
                (n_samples, 2 * n_epochs).

        Returns:
            torch.Tensor: Tensor of processed results. This tensor is of shape
                (n_samples, 2 * n_epochs).
        """

        return results / self.scale
    
    def get_scale(self):
        return self.scale
