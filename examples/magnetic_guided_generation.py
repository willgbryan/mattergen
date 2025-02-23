"""
Example script demonstrating magnetic property-guided generation with MatterGen.

This script shows how to:
1. Set up a base MatterGen generator
2. Add magnetic property guidance
3. Generate structures with target magnetic properties
4. Evaluate the magnetic properties of generated structures
"""

import os
from pathlib import Path
import torch
import zipfile
import tempfile
from chgnet.model.model import CHGNet
from mattergen.generator import CrystalGenerator
from mattergen.diffusion.sampling.reward_functions import MagneticRewardFunction
from mattergen.utils.magnetic_guidance import create_property_guided_generator
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.diffusion.sampling.reward_functions import BaseRewardFunction

class LoggingMagneticDensityRewardFunction(BaseRewardFunction):
    """Magnetic density reward function with detailed logging."""
    
    def __init__(self, target_magnetic_density: float, chgnet_model=None):
        """
        Args:
            target_magnetic_density: Target magnetic density (in μB/Å³) to guide towards
            chgnet_model: Pre-loaded CHGNet model, will load default if None
        """
        from chgnet.model.model import CHGNet
        self.target_magnetic_density = target_magnetic_density
        self.chgnet_model = chgnet_model or CHGNet.load()
        
        # Constants for reward scaling
        self.density_weight = 1.0
        self.moment_weight = 0.3
        self.volume_weight = 0.2
        self.target_volume = 200.0  # Å³, reasonable target for unit cell volume
    
    def compute_reward(self, batch: dict) -> torch.Tensor:
        structures = self._batch_to_structures(batch)
        rewards = []
        
        print("\nComputing magnetic density rewards for batch:")
        for i, structure in enumerate(structures):
            prediction = self.chgnet_model.predict_structure(structure)
            mag_moments = prediction['m']  # Per-atom array
            mag_moment = float(sum(mag_moments))  # Total magnetic moment
            volume = structure.volume
            mag_density = mag_moment / volume
            
            # Compute multi-term reward
            # 1. Density matching term (exponential to make it stronger near target)
            density_reward = -torch.exp(torch.tensor(abs(mag_density - self.target_magnetic_density)))
            
            # 2. Moment encouragement term (using tanh for smooth scaling)
            moment_reward = torch.tanh(torch.tensor(mag_moment / 10.0))  # Scale by 10 for reasonable tanh range
            
            # 3. Volume control term (quadratic with minimum at target_volume)
            volume_diff = (volume - self.target_volume) / self.target_volume
            volume_reward = -volume_diff * volume_diff
            
            # Combine rewards with weights
            reward = (self.density_weight * density_reward + 
                     self.moment_weight * moment_reward +
                     self.volume_weight * volume_reward)
            
            print(f"  Structure {i}:")
            print(f"    Volume: {volume:.3f} Å³")
            print(f"    Total magnetic moment: {mag_moment:.3f} μB")
            print(f"    Magnetic density: {mag_density:.6f} μB/Å³")
            print(f"    Target density: {self.target_magnetic_density:.6f} μB/Å³")
            print(f"    Density reward: {density_reward:.3f}")
            print(f"    Moment reward: {moment_reward:.3f}")
            print(f"    Volume reward: {volume_reward:.3f}")
            print(f"    Total reward: {reward:.3f}")
            
            rewards.append(reward)
            
        return torch.tensor(rewards, device=batch['pos'].device)

def main():
    # Configuration
    BASE_RESULTS_PATH = Path("/teamspace/studios/this_studio/mattergen/results")
    RESULTS_PATH = BASE_RESULTS_PATH / "magnetic_guided"
    TARGET_MAGNETIC_DENSITY = 1.5  # Target magnetic density in μB/Å³
    GUIDANCE_SCALE = 2.0  # Increased from 1.0 to give stronger guidance
    BATCH_SIZE = 8
    NUM_BATCHES = 2
    
    # Create results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("Setting up models...")
    
    # Load CHGNet for magnetic property prediction
    chgnet = CHGNet.load()
    
    # Create base MatterGen generator
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        record_trajectories=True
    )
    
    print(f"Creating magnetically-guided generator (target density: {TARGET_MAGNETIC_DENSITY} μB/Å³)...")
    
    # Add magnetic property guidance with logging
    guided_generator = create_property_guided_generator(
        generator=generator,
        reward_function=LoggingMagneticDensityRewardFunction(
            target_magnetic_density=TARGET_MAGNETIC_DENSITY,
            chgnet_model=chgnet
        ),
        guidance_scale=GUIDANCE_SCALE
    )
    
    print("Generating structures...")
    
    # Generate structures
    guided_generator.generate(output_dir=RESULTS_PATH)
    
    print("\nEvaluating generated structures...")
    
    # Look for the zip file containing generated structures
    zip_file = RESULTS_PATH / "generated_crystals_cif.zip"
    
    if not zip_file.exists():
        print(f"No zip file found at {zip_file}")
        print("Checking alternative locations...")
        alt_paths = [
            BASE_RESULTS_PATH / "multi_property_guided" / "generated_crystals_cif.zip",
            Path("results/magnetic_guided/generated_crystals_cif.zip"),
            Path("results/multi_property_guided/generated_crystals_cif.zip")
        ]
        for path in alt_paths:
            if path.exists():
                zip_file = path
                print(f"Found zip file at: {path}")
                break
        else:
            print("No structures were generated! Check the generation process.")
            return
    
    # Create a temporary directory to extract CIF files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nExtracting CIF files from {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Load and evaluate extracted structures
        from pymatgen.core import Structure
        cif_files = list(Path(temp_dir).glob("*.cif"))
        print(f"\nFound {len(cif_files)} CIF files")
        
        if not cif_files:
            print("No structures were found in the zip file!")
            return
        
        structures = []
        for f in cif_files:
            try:
                structure = Structure.from_file(f)
                structures.append(structure)
            except Exception as e:
                print(f"Error loading structure from {f}: {e}")
        
        print(f"Successfully loaded {len(structures)} structures")
        
        if not structures:
            print("No valid structures were loaded! Check the CIF files.")
            return
        
        # Compute magnetic moments and densities
        moments = []
        densities = []
        for i, structure in enumerate(structures):
            try:
                prediction = chgnet.predict_structure(structure)
                mag_moment = float(sum(prediction['m']))  # Total magnetic moment
                volume = structure.volume
                mag_density = mag_moment / volume
                
                moments.append(mag_moment)
                densities.append(mag_density)
                
                print(f"\nStructure {i} final evaluation:")
                print(f"  Volume: {volume:.3f} Å³")
                print(f"  Total magnetic moment: {mag_moment:.3f} μB")
                print(f"  Magnetic density: {mag_density:.6f} μB/Å³")
                print(f"  Per-atom moments: {prediction['m']}")
            except Exception as e:
                print(f"Error evaluating structure {i}: {e}")
        
        if not moments:
            print("No valid magnetic moment predictions! Check the CHGNet evaluation.")
            return
        
        # Print magnetic property statistics
        moments = torch.tensor(moments)
        densities = torch.tensor(densities)
        print("\nMagnetic Properties:")
        print(f"Average magnetic moment: {moments.mean():.2f} μB")
        print(f"Std of magnetic moments: {moments.std():.2f} μB")
        print(f"Min/Max magnetic moment: {moments.min():.2f}/{moments.max():.2f} μB")
        print(f"\nMagnetic Density Properties:")
        print(f"Average magnetic density: {densities.mean():.6f} μB/Å³")
        print(f"Std of magnetic densities: {densities.std():.6f} μB/Å³")
        print(f"Min/Max magnetic density: {densities.min():.6f}/{densities.max():.6f} μB/Å³")
        
        print(f"\nResults saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
