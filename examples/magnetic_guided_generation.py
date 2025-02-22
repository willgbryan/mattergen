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
from chgnet.model.model import CHGNet
from mattergen.generator import CrystalGenerator
from mattergen.diffusion.sampling.reward_functions import MagneticRewardFunction
from mattergen.utils.magnetic_guidance import create_property_guided_generator

def main():
    # Configuration
    RESULTS_PATH = Path("results/magnetic_guided")
    TARGET_MAGNETIC_MOMENT = 2.0  # Target magnetic moment in μB
    GUIDANCE_SCALE = 1.0  # How strongly to enforce the magnetic property
    BATCH_SIZE = 8
    NUM_BATCHES = 2
    
    # Create results directory
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    print("Setting up models...")
    
    # Load CHGNet for magnetic property prediction
    chgnet = CHGNet.load()
    
    # Create base MatterGen generator
    generator = CrystalGenerator(
        pretrained_name="mattergen_base",  # Use pre-trained base model
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        record_trajectories=True  # Record the denoising trajectory
    )
    
    print(f"Creating magnetically-guided generator (target moment: {TARGET_MAGNETIC_MOMENT} μB)...")
    
    # Add magnetic property guidance
    guided_generator = create_property_guided_generator(
        generator=generator,
        reward_function=MagneticRewardFunction(
            target_magnetic_moment=TARGET_MAGNETIC_MOMENT,
            chgnet_model=chgnet
        ),
        guidance_scale=GUIDANCE_SCALE
    )
    
    print("Generating structures...")
    
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
    
    # Compute magnetic moments
    moments = []
    for structure in structures:
        prediction = chgnet.predict_structure(structure)
        mag_moment = prediction['magmom'].mean()
        moments.append(mag_moment)
    
    # Print statistics
    moments = torch.tensor(moments)
    print(f"\nGenerated {len(structures)} structures")
    print(f"Average magnetic moment: {moments.mean():.2f} μB")
    print(f"Std of magnetic moments: {moments.std():.2f} μB")
    print(f"Min magnetic moment: {moments.min():.2f} μB")
    print(f"Max magnetic moment: {moments.max():.2f} μB")
    
    print(f"\nResults saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
