"""
Example script demonstrating multi-property guided generation with MatterGen.

This script shows how to:
1. Create multiple reward functions
2. Combine them with different weights
3. Generate structures guided by multiple property targets
"""

import os
from pathlib import Path
import torch
from chgnet.model.model import CHGNet
from mattergen.generator import CrystalGenerator
from mattergen.diffusion.sampling.reward_functions import (
    BaseRewardFunction,
    MagneticRewardFunction,
    CompositeRewardFunction
)
from mattergen.utils.magnetic_guidance import create_property_guided_generator

class StabilityRewardFunction(BaseRewardFunction):
    """Example reward function for stability using formation energy."""
    
    def __init__(self, max_formation_energy: float = 0.1):
        self.max_formation_energy = max_formation_energy
        self.chgnet = CHGNet.load()
    
    def compute_reward(self, batch: dict) -> torch.Tensor:
        structures = self._batch_to_structures(batch)
        rewards = []
        
        for structure in structures:
            prediction = self.chgnet.predict_structure(structure)
            # Assume formation_energy is in eV/atom
            formation_energy = prediction['energy'] / len(structure)
            
            # Reward is higher for more stable structures
            # Scale to roughly match magnitude of other rewards
            reward = -5.0 * abs(formation_energy)
            if formation_energy > self.max_formation_energy:
                reward *= 2.0  # Additional penalty for unstable structures
                
            rewards.append(reward)
            
        return torch.tensor(rewards, device=batch['pos'].device)

def main():
    # Configuration
    RESULTS_PATH = Path("results/multi_property_guided")
    BATCH_SIZE = 8
    NUM_BATCHES = 2
    
    # Property targets
    TARGET_MAGNETIC_MOMENT = 2.0  # Target magnetic moment in μB
    MAX_FORMATION_ENERGY = 0.1  # Maximum formation energy in eV/atom
    
    # Guidance parameters
    GUIDANCE_SCALE = 1.0
    REWARD_WEIGHTS = {
        'magnetic': 0.7,  # Prioritize magnetic properties
        'stability': 0.3  # Secondary focus on stability
    }
    
    # Create results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("Setting up models...")
    
    # Load CHGNet (will be shared between reward functions)
    chgnet = CHGNet.load()
    
    # Create reward functions
    reward_functions = {
        'magnetic': MagneticRewardFunction(
            target_magnetic_moment=TARGET_MAGNETIC_MOMENT,
            chgnet_model=chgnet
        ),
        'stability': StabilityRewardFunction(
            max_formation_energy=MAX_FORMATION_ENERGY
        )
    }
    
    # Create base generator
    generator = CrystalGenerator(
        pretrained_name="mattergen_base",
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        record_trajectories=True
    )
    
    print("Creating multi-property guided generator...")
    print(f"- Target magnetic moment: {TARGET_MAGNETIC_MOMENT} μB")
    print(f"- Max formation energy: {MAX_FORMATION_ENERGY} eV/atom")
    print(f"- Reward weights: {REWARD_WEIGHTS}")
    
    # Add multi-property guidance
    guided_generator = create_property_guided_generator(
        generator=generator,
        reward_function=reward_functions,
        reward_weights=REWARD_WEIGHTS,
        guidance_scale=GUIDANCE_SCALE
    )
    
    print("\nGenerating structures...")
    
    # Generate structures
    guided_generator.generate(
        results_path=RESULTS_PATH,
        save_trajectories=True
    )
    
    print("\nEvaluating generated structures...")
    
    # Load and evaluate generated structures
    from pymatgen.core import Structure
    structures = [
        Structure.from_file(f) 
        for f in RESULTS_PATH.glob("*.cif")
    ]
    
    # Compute properties
    moments = []
    energies = []
    for structure in structures:
        prediction = chgnet.predict_structure(structure)
        moments.append(prediction['magmom'].mean())
        energies.append(prediction['energy'] / len(structure))
    
    # Print magnetic moment statistics
    moments = torch.tensor(moments)
    print("\nMagnetic Properties:")
    print(f"Average magnetic moment: {moments.mean():.2f} μB")
    print(f"Std of magnetic moments: {moments.std():.2f} μB")
    print(f"Min/Max magnetic moment: {moments.min():.2f}/{moments.max():.2f} μB")
    
    # Print formation energy statistics
    energies = torch.tensor(energies)
    print("\nStability Properties:")
    print(f"Average formation energy: {energies.mean():.3f} eV/atom")
    print(f"Std of formation energies: {energies.std():.3f} eV/atom")
    print(f"Min/Max formation energy: {energies.min():.3f}/{energies.max():.3f} eV/atom")
    
    print(f"\nResults saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
