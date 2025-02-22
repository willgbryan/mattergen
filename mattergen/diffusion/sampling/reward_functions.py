from abc import ABC, abstractmethod
import torch
from typing import Dict, Any
from pymatgen.core import Structure

class BaseRewardFunction(ABC):
    """
    Abstract base class for reward functions used in guided generation.
    
    Subclass this to implement custom reward functions for property-guided generation.
    The reward function should return higher values for more desirable structures.
    """
    
    @abstractmethod
    def compute_reward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute reward values for a batch of structures.
        
        Args:
            batch: Dictionary containing structure information with keys:
                - pos: Atomic positions (batch_size, num_atoms, 3)
                - cell: Unit cell parameters (batch_size, 3, 3)
                - atomic_numbers: Atomic numbers (batch_size, num_atoms)
                
        Returns:
            Tensor of shape (batch_size,) containing reward values
        """
        pass
    
    def _batch_to_structures(self, batch: Dict[str, torch.Tensor]) -> list[Structure]:
        """
        Convert a batch dictionary to a list of pymatgen Structure objects.
        
        Args:
            batch: Dictionary containing pos, cell, atomic_numbers
            
        Returns:
            List of Structure objects
        """
        structures = []
        pos = batch['pos'].detach().cpu().numpy()
        cell = batch['cell'].detach().cpu().numpy()
        atomic_numbers = batch['atomic_numbers'].detach().cpu().numpy()
        
        for i in range(len(pos)):
            structure = Structure(
                lattice=cell[i],
                species=atomic_numbers[i],
                coords=pos[i],
                coords_are_cartesian=True
            )
            structures.append(structure)
            
        return structures


class MagneticRewardFunction(BaseRewardFunction):
    """
    Reward function for magnetic property guidance using CHGNet.
    """
    
    def __init__(self, target_magnetic_moment: float, chgnet_model=None):
        """
        Args:
            target_magnetic_moment: Target magnetic moment (in μB) to guide towards
            chgnet_model: Pre-loaded CHGNet model, will load default if None
        """
        from chgnet.model.model import CHGNet
        
        self.target_magnetic_moment = target_magnetic_moment
        self.chgnet_model = chgnet_model or CHGNet.load()
        
    def compute_reward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute reward based on magnetic moment prediction from CHGNet.
        """
        structures = self._batch_to_structures(batch)
        
        rewards = []
        for structure in structures:
            prediction = self.chgnet_model.predict_structure(structure)
            mag_moment = prediction['magmom'].mean()
            
            # Compute reward as negative absolute difference from target
            reward = -abs(mag_moment - self.target_magnetic_moment)
            rewards.append(reward)
            
        return torch.tensor(rewards, device=batch['pos'].device)


class CompositeRewardFunction(BaseRewardFunction):
    """
    Combines multiple reward functions with optional weights.
    """
    
    def __init__(self, reward_functions: Dict[str, BaseRewardFunction], weights: Dict[str, float] = None):
        """
        Args:
            reward_functions: Dictionary mapping names to reward functions
            weights: Optional dictionary of weights for each reward function
        """
        self.reward_functions = reward_functions
        self.weights = weights or {name: 1.0 for name in reward_functions}
        
    def compute_reward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted sum of rewards from all reward functions.
        """
        total_reward = 0
        for name, func in self.reward_functions.items():
            reward = func.compute_reward(batch)
            total_reward += self.weights[name] * reward
        return total_reward
