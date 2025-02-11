import math
import sys
import warnings
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import grad

from tqdm import tqdm

from ..common import utils

class MHBase(nn.Module):
    """
    A base class for implementing Metropolis-Hastings (MH) sampling methods.

    This class provides a base implementation for collecting samples using the Metropolis-Hastings (MH) algorithm.
    Subclasses can inherit from this base class and override the `mh_engine` method to implement specific MH sampling
    engines.

    Methods:
        collect_samples(n_steps=1, n_burnin=0, keep_burnin=False):
            Collect samples using the MH algorithm.

    Attributes:
        mh_engine (callable): A callable method to perform MH sampling steps.
        params: Model parameters used in the MH sampling.
        energy_fn: Energy function used for evaluating the energy of states.
        mh_kwargs (dict): Additional keyword arguments for the MH sampling engine.
    """
    
    def __init__(self):
        """
        Initialize the MHBase class.
        """
        super().__init__()
        
    def collect_samples(self, n_steps=1, n_burnin=0, keep_burnin=False):
        """
        Collect samples using the Metropolis-Hastings (MH) algorithm.

        Args:
            n_steps (int): The number of MH sampling steps to perform.
            n_burnin (int): The number of burn-in steps to perform before collecting samples.
            keep_burnin (bool): Whether to keep the burn-in samples.

        Returns:
            dict: A dictionary containing collected samples and, optionally, burn-in samples.
                  The dictionary has the keys 'burnin' and 'samples'.
                  Each of these keys maps to a sub-dictionary with keys 'states', 'energies', and 'acceptances'.
                  The values associated with these keys are torch tensors containing the collected data.
        """
        burnin = None
        samples= None
        
        if n_burnin >= 1:
            print('burn in', file=sys.stderr)
            burnin  = {'states':[], 'energies': [], 'acceptances': []}
            for t in tqdm(range(n_burnin)):
                sample = self.mh_engine(self.params, self.energy_fn, **self.mh_kwargs)
                if keep_burnin:
                    burnin['states'].append(sample['state'])
                    burnin['energies'].append(sample['energy'])
                    burnin['acceptances'].append(sample['acceptance'])
                
        if keep_burnin:
            burnin = { k: torch.stack(v, dim=0) for k,v in burnin.items() }
    
        if n_steps >= 1:
            print('collect samples', file=sys.stderr)
            samples = {'samples':[], 'energies': [], 'acceptances': []}
            for t in tqdm(range(n_steps)):
                sample = self.mh_engine(self.params, self.energy_fn, **self.mh_kwargs)
                samples['states'].append(sample['state'])
                samples['energies'].append(sample['energy'])
                samples['acceptances'].append(sample['acceptance'])

        samples = { k: torch.stack(v, dim=0) for k,v in samples.items() }
        
        return {'burnin': burnin, 'samples':samples}

@torch.no_grad()
def naive_mh_step(params, energy_fn, n_positions=1, temperature=1.0):
    """
    Perform a single step of the Naive Metropolis-Hastings (MH) sampling algorithm.

    Args:
        params (nn.Module): The model parameters.
        energy_fn (callable): The energy function used to evaluate the energy of states.
        n_positions (int): The number of positions to propose updates for.
        temperature (float): The temperature parameter for MH sampling.

    Returns:
        dict: A dictionary containing information about the MH sampling step.
            The dictionary has the keys 'state', 'energy', and 'acceptance'.
            'state': A tensor representing the proposed state after the MH step.
            'energy': A tensor containing the energy of the proposed state.
            'acceptance': A boolean tensor indicating whether the proposed state was accepted.
    """
    assert len(params.theta.shape) == 3
    
    old_params = params.theta.detach().clone()
    old_seq    = params()
    old_energy = energy_fn(old_seq)
    old_energy = params.rebatch( old_energy )
    old_nll = old_params * -115
    
    pos_shuffle = torch.argsort( torch.rand(old_params.shape[0], old_params.shape[-1]), dim=-1 )
    proposed_positions = pos_shuffle[:,:n_positions]
    batch_slicer = torch.arange(old_params.shape[0]).view(-1,1)
    
    updates = old_nll[batch_slicer, :, proposed_positions].mul(-1)
    old_nll[batch_slicer, :, proposed_positions] = updates
    
    proposal_dist = dist.OneHotCategorical(logits=-old_nll.permute(0,2,1)/temperature)
    
    new_params = proposal_dist.sample().permute(0,2,1).detach().clone()
    params.theta.data = new_params # temporary update
    new_seq    = params()
    new_energy = energy_fn(new_seq)
    new_energy = params.rebatch( new_energy )
    
    u = torch.rand_like(old_energy).log()
    accept = u.le( (old_energy-new_energy)/temperature )
    
    # sample = torch.stack([old_params, new_params], dim=0)[accept.long(),torch.arange(accept.numel())].detach().clone()
    sample = old_params.clone()
    sample[accept.squeeze(-1)] = new_params[accept.squeeze(-1)]
    # energy = torch.stack([old_energy, new_energy], dim=0)[accept.long(),torch.arange(accept.numel())].detach().clone()
    energy = old_energy.clone()
    energy[accept.squeeze(-1)] = new_energy[accept.squeeze(-1)]
    
    # params.theta.data = sample # metropolis corrected update
    params.theta.data = sample.view(old_params.shape)
    
    return {'state': sample.detach().clone().cpu(), 
            'energy': energy.detach().clone().cpu(), 
            'acceptance': accept.detach().clone().cpu()}

class NaiveMH(MHBase):
    """
    Implementation of the Naive Metropolis-Hastings (MH) sampling algorithm.

    Args:
        energy_fn (callable): The energy function used to evaluate the energy of states.
        params (nn.Module): The model parameters.
        n_positions (int): The number of positions to propose updates for.
        temperature (float): The temperature parameter for MH sampling.

    Inherits from:
        MHBase (nn.Module): Base class for Metropolis-Hastings samplers.

    Attributes:
        energy_fn (callable): The energy function used to evaluate the energy of states.
        params (nn.Module): The model parameters.
        n_positions (int): The number of positions to propose updates for.
        temperature (float): The temperature parameter for MH sampling.
        mh_kwargs (dict): Keyword arguments for the MH sampling engine.
        mh_engine (function): The MH sampling engine (naive_mh_step function).

    Methods:
        collect_samples(n_steps=1, n_burnin=0, keep_burnin=False):
            Collect samples using the Naive Metropolis-Hastings algorithm.

    """
    
    def __init__(self, 
                 energy_fn, 
                 params,
                 n_positions=1, 
                 temperature=1.0
                ):
        """
        Initialize the NaiveMH class.

        Args:
            energy_fn (callable): The energy function used to evaluate the energy of states.
            params (nn.Module): The model parameters.
            n_positions (int): The number of positions to propose updates for.
            temperature (float): The temperature parameter for MH sampling.
        """
        super().__init__()
        self.energy_fn = energy_fn
        self.params = params
        self.n_positions = n_positions
        self.temperature = temperature
        
        self.mh_kwargs = {'n_positions': self.n_positions, 
                          'temperature': self.temperature}
        
        self.mh_engine = naive_mh_step
        
class PolynomialDecay:
    """
    Polynomial decay schedule.

    Args:
        a (float): Coefficient a in the polynomial decay equation.
        b (float): Coefficient b in the polynomial decay equation.
        gamma (float): Exponent gamma in the polynomial decay equation.

    Methods:
        __call__():
            Calculate the current decay value based on the polynomial decay equation.
        step():
            Advance the decay step and return the updated decay value.
        reset():
            Reset the decay step to the initial state.

    """
    
    def __init__(self,
                 a = 1,
                 b = 1,
                 gamma = 1.,
                ):
        """
        Initialize the PolynomialDecay schedule.

        Args:
            a (float): Coefficient a in the polynomial decay equation.
            b (float): Coefficient b in the polynomial decay equation.
            gamma (float): Exponent gamma in the polynomial decay equation.
        """
        self.a = a
        self.b = b
        self.gamma = gamma
        self.t = 0
        
    def __call__(self):
        """
        Calculate the current decay value based on the polynomial decay equation.

        Returns:
            float: The current decay value.
        """
        return self.a*((self.b+self.t)**-self.gamma)
    
    def step(self):
        """
        Advance the decay step and return the updated decay value.

        Returns:
            float: The updated decay value.
        """
        val = self()
        self.t += 1
        return val
    
    def reset(self):
        """
        Reset the decay step to the initial state.

        Returns:
            None
        """
        self.t = 0
        return None

class SimulatedAnnealing(nn.Module):
    """
    Simulated Annealing generator using Metropolis-Hastings sampling.

    Args:
        params (nn.Module): The parameterized distribution to sample from.
        energy_fn (callable): Energy function to calculate the energy of a sample.
        n_positions (int, optional): Number of positions to update per step. Default is 1.
        a (float, optional): Coefficient a in the polynomial decay equation. Default is 1.
        b (float, optional): Coefficient b in the polynomial decay equation. Default is 1.
        gamma (float, optional): Exponent gamma in the polynomial decay equation. Default is 1.

    Methods:
        collect_samples(n_steps=1, n_burnin=0, keep_burnin=False):
            Collect Metropolis-Hastings samples using simulated annealing.
        generate(n_proposals=1, energy_threshold=float("Inf"), max_attempts=10000, n_steps=1, n_burnin=0, keep_burnin=False):
            Generate proposals using simulated annealing and Metropolis-Hastings sampling.

    """
    
    @staticmethod
    def add_generator_specific_args(parent_parser):
        """
        Add generator-specific arguments to an existing argument parser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added generator-specific arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        
        group  = parser.add_argument_group('Generator Constructor args')
        group.add_argument('--n_positions', type=int, default=1)
        group.add_argument('--a', type=float, default=1.)
        group.add_argument('--b', type=float, default=1.)
        group.add_argument('--gamma', type=float, default=1.)
        
        group  = parser.add_argument_group('Generator Runtime args')
        group.add_argument('--n_steps', type=int, default=1)
        group.add_argument('--n_burnin', type=int, default=0)
        group.add_argument('--keep_burnin', type=utils.str2bool, default=False)
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Process grouped arguments.

        Args:
            grouped_args (dict): Grouped arguments.

        Returns:
            tuple: Tuple containing constructor arguments and runtime arguments.
        """
        constructor_args = grouped_args['Generator Constructor args']
        runtime_args     = grouped_args['Generator Runtime args']
        
        return constructor_args, runtime_args
    
    def __init__(self, 
                 params,
                 energy_fn, 
                 n_positions=1, 
                 a=1.,
                 b=1.,
                 gamma=1.,
                ):
        """
        Initialize the SimulatedAnnealing generator.

        Args:
            params (nn.Module): The parameterized distribution to sample from.
            energy_fn (callable): Energy function to calculate the energy of a sample.
            n_positions (int, optional): Number of positions to update per step. Default is 1.
            a (float, optional): Coefficient a in the polynomial decay equation. Default is 1.
            b (float, optional): Coefficient b in the polynomial decay equation. Default is 1.
            gamma (float, optional): Exponent gamma in the polynomial decay equation. Default is 1.
        """
        super().__init__()
        self.params = params
        self.energy_fn = energy_fn
        self.n_positions = n_positions
        self.a = a
        self.b = b
        self.gamma = gamma
        self.temperature_schedule = PolynomialDecay(a,b,gamma)
        
        self.mh_engine = naive_mh_step

    def collect_samples(self, n_steps=1, n_burnin=0, keep_burnin=False):
        """
        Collect Metropolis-Hastings samples using simulated annealing.

        Args:
            n_steps (int, optional): Number of steps to collect samples. Default is 1.
            n_burnin (int, optional): Number of burn-in steps. Default is 0.
            keep_burnin (bool, optional): Whether to keep burn-in samples. Default is False.

        Returns:
            dict: Dictionary containing burn-in and sample trajectories.
        """
        burnin = None
        samples= None
        
        if n_burnin >= 1:
            print('burn in', file=sys.stderr)
            burnin  = {'states':[], 'energies': [], 'acceptances': []}
            self.temperature_schedule.reset()
            for t in tqdm(range(n_burnin)):
                temp = self.temperature_schedule.step()
                sample = self.mh_engine(self.params, self.energy_fn, n_positions=self.n_positions, temperature=temp)
                if keep_burnin:
                    burnin['states'].append(sample['state'])
                    burnin['energies'].append(sample['energy'])
                    burnin['acceptances'].append(sample['acceptance'])
                
        if keep_burnin:
            burnin = { k: torch.stack(v, dim=0) for k,v in burnin.items() }
    
        if n_steps >= 1:
            print("step0", n_steps)
            print('collect samples', file=sys.stderr)
            samples = {'states':[], 'energies': [], 'acceptances': []}
            self.temperature_schedule.reset()
            for t in tqdm(range(n_steps)):
                temp = self.temperature_schedule.step()
                sample = self.mh_engine(self.params, self.energy_fn, n_positions=self.n_positions, temperature=temp)
                samples['states'].append(sample['state'])
                samples['energies'].append(sample['energy'])
                samples['acceptances'].append(sample['acceptance'])

        samples = { k: torch.stack(v, dim=0) for k,v in samples.items() }
        
        return {'burnin': burnin, 'samples':samples}

    def generate(self, n_proposals=1, energy_threshold=float("Inf"), max_attempts=10000, 
                 n_steps=1, n_burnin=0, keep_burnin=False):
        """
        Generate proposals using simulated annealing and Metropolis-Hastings sampling.

        Args:
            n_proposals (int, optional): Number of proposals to generate. Default is 1.
            energy_threshold (float, optional): Energy threshold for proposal acceptance. Default is float("Inf").
            max_attempts (int, optional): Maximum number of attempts to generate proposals. Default is 10000.
            n_steps (int, optional): Number of steps for each proposal generation. Default is 1.
            n_burnin (int, optional): Number of burn-in steps for each proposal generation. Default is 0.
            keep_burnin (bool, optional): Whether to keep burn-in samples for each proposal. Default is False.

        Returns:
            dict: Dictionary containing generated proposals, energies, and acceptance rate.
        """
        batch_size, *theta_shape = self.params.theta.shape
        proposals = torch.randn([0,*theta_shape])
        energies  = torch.randn([0])
        
        acceptance = torch.randn([0])
        
        attempts = 0
        
        while (proposals.shape[0] < n_proposals) and (attempts < max_attempts):
            
            attempts += 1
            
            trajectory = self.collect_samples(
                n_steps=n_steps, n_burnin=n_burnin, keep_burnin=keep_burnin
            )
            
            final_states  = trajectory['samples']['states'][-1]
            self.params.theta.data = final_states.to(self.params.theta.device)
            final_energies = self.energy_fn.energy_calc( self.params() )
            final_energies = self.params.rebatch( final_energies ) \
                               .detach().clone().cpu()
            
            energy_filter = final_energies <= energy_threshold
            
            # proposals = torch.cat([proposals,  final_states[energy_filter]], dim=0)
            # energies  = torch.cat([energies, final_energies[energy_filter]], dim=0)
            proposals = torch.cat([proposals, final_states[energy_filter.squeeze(-1)]], dim=0)
            energies  = torch.cat([energies, final_energies[energy_filter.squeeze(-1)]], dim=0)
            
            print(f'attempt {attempts} acceptance rate: {energy_filter.sum().item()}/{energy_filter.numel()}')
            
            if acceptance is None:
                acceptance = energy_filter.float().mean().item()
            else:
                ((attempts/(attempts+1))*acceptance) + ((1/(attempts+1))*energy_filter.float().mean().item())
                        
            try:
                self.params.reset()
            except NotImplementedError:
                pass
            
        return {'proposals': proposals[:n_proposals], 'energies': energies[:n_proposals], 'acceptance_rate': acceptance}


