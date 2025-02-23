"""
Hyperparameter tuning script for magnetic density optimization in MatterGen.

This script implements Phase 1 of the optimization strategy:
1. Grid search over core parameters
2. Metrics collection and logging
3. CSV output for analysis
"""

import os
import sys
import json
import time
import argparse
import itertools
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import zipfile
import tempfile
from chgnet.model.model import CHGNet
from mattergen.generator import CrystalGenerator
from mattergen.diffusion.sampling.reward_functions import BaseRewardFunction
from mattergen.utils.magnetic_guidance import create_property_guided_generator
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from pymatgen.core import Structure

class ConfigurableMagneticDensityReward(BaseRewardFunction):
    """Configurable magnetic density reward function for hyperparameter tuning."""
    
    def __init__(
        self,
        target_magnetic_density: float,
        density_weight: float = 1.0,
        moment_weight: float = 0.3,
        volume_weight: float = 0.2,
        target_volume: float = 200.0,
        chgnet_model=None
    ):
        """
        Args:
            target_magnetic_density: Target magnetic density (in μB/Å³)
            density_weight: Weight for density matching term
            moment_weight: Weight for moment encouragement term
            volume_weight: Weight for volume control term
            target_volume: Target volume in Å³
            chgnet_model: Pre-loaded CHGNet model
        """
        from chgnet.model.model import CHGNet
        self.target_magnetic_density = target_magnetic_density
        self.density_weight = density_weight
        self.moment_weight = moment_weight
        self.volume_weight = volume_weight
        self.target_volume = target_volume
        self.chgnet_model = chgnet_model or CHGNet.load()
        
        # For logging
        self.current_rewards = []
        self.current_densities = []
        self.current_moments = []
        self.current_volumes = []
    
    def compute_reward(self, batch: dict) -> torch.Tensor:
        structures = self._batch_to_structures(batch)
        rewards = []
        
        # Clear previous batch metrics
        self.current_rewards = []
        self.current_densities = []
        self.current_moments = []
        self.current_volumes = []
        
        print("\nComputing magnetic density rewards for batch:")
        for i, structure in enumerate(structures):
            prediction = self.chgnet_model.predict_structure(structure)
            mag_moments = prediction['m']  # Per-atom array
            mag_moment = float(sum(mag_moments))  # Total magnetic moment
            volume = structure.volume
            mag_density = mag_moment / volume
            
            # Compute multi-term reward
            # 1. Density matching term (exponential)
            density_reward = -torch.exp(torch.tensor(abs(mag_density - self.target_magnetic_density)))
            
            # 2. Moment encouragement term (tanh)
            moment_reward = torch.tanh(torch.tensor(mag_moment / 10.0))
            
            # 3. Volume control term (quadratic)
            volume_diff = (volume - self.target_volume) / self.target_volume
            volume_reward = -volume_diff * volume_diff
            
            # Combine rewards with weights
            reward = (self.density_weight * density_reward + 
                     self.moment_weight * moment_reward +
                     self.volume_weight * volume_reward)
            
            # Store metrics for logging
            self.current_rewards.append(float(reward))
            self.current_densities.append(mag_density)
            self.current_moments.append(mag_moment)
            self.current_volumes.append(volume)
            
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

def evaluate_structures(zip_file: Path, chgnet_model=None) -> dict:
    """Evaluate structures from a zip file and compute metrics."""
    if not zip_file.exists():
        raise FileNotFoundError(f"No zip file found at {zip_file}")
    
    # Load CHGNet if not provided
    if chgnet_model is None:
        chgnet_model = CHGNet.load()
    
    metrics = {
        'moments': [],
        'densities': [],
        'volumes': [],
        'num_atoms': []
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        cif_files = list(Path(temp_dir).glob("*.cif"))
        if not cif_files:
            raise ValueError("No structures found in zip file")
        
        for f in cif_files:
            try:
                structure = Structure.from_file(f)
                prediction = chgnet_model.predict_structure(structure)
                
                mag_moment = float(sum(prediction['m']))
                volume = structure.volume
                mag_density = mag_moment / volume
                
                metrics['moments'].append(mag_moment)
                metrics['densities'].append(mag_density)
                metrics['volumes'].append(volume)
                metrics['num_atoms'].append(len(structure))
                
            except Exception as e:
                print(f"Error evaluating structure from {f}: {e}")
    
    if not metrics['moments']:
        raise ValueError("No valid structures were evaluated")
    
    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    # Compute summary statistics
    summary = {
        'avg_density': float(np.mean(metrics['densities'])),
        'std_density': float(np.std(metrics['densities'])),
        'max_density': float(np.max(metrics['densities'])),
        'min_density': float(np.min(metrics['densities'])),
        'avg_moment': float(np.mean(metrics['moments'])),
        'std_moment': float(np.std(metrics['moments'])),
        'avg_volume': float(np.mean(metrics['volumes'])),
        'std_volume': float(np.std(metrics['volumes'])),
        'avg_num_atoms': float(np.mean(metrics['num_atoms'])),
        'pct_above_0.1_target': float(np.mean(metrics['densities'] > 0.15)),  # 10% of target
        'pct_above_0.5_target': float(np.mean(metrics['densities'] > 0.75)),  # 50% of target
        'num_structures': len(metrics['densities'])
    }
    
    return summary

def run_trial(
    trial_id: int,
    results_dir: Path,
    guidance_scale: float,
    density_weight: float,
    moment_weight: float,
    volume_weight: float,
    target_volume: float,
    target_density: float = 1.5,
    batch_size: int = 4,
    num_batches: int = 1
) -> dict:
    """Run a single trial with specified hyperparameters."""
    
    trial_dir = results_dir / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    params = {
        'trial_id': trial_id,
        'guidance_scale': guidance_scale,
        'density_weight': density_weight,
        'moment_weight': moment_weight,
        'volume_weight': volume_weight,
        'target_volume': target_volume,
        'target_density': target_density,
        'batch_size': batch_size,
        'num_batches': num_batches,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(trial_dir / "params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    try:
        # Load models
        chgnet = CHGNet.load()
        
        # Create generator
        generator = CrystalGenerator(
            checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
            batch_size=batch_size,
            num_batches=num_batches,
            record_trajectories=True
        )
        
        # Create reward function
        reward_fn = ConfigurableMagneticDensityReward(
            target_magnetic_density=target_density,
            density_weight=density_weight,
            moment_weight=moment_weight,
            volume_weight=volume_weight,
            target_volume=target_volume,
            chgnet_model=chgnet
        )
        
        # Add guidance
        guided_generator = create_property_guided_generator(
            generator=generator,
            reward_function=reward_fn,
            guidance_scale=guidance_scale
        )
        
        # Generate structures
        guided_generator.generate(output_dir=trial_dir)
        
        # Evaluate results
        zip_file = trial_dir / "generated_crystals_cif.zip"
        metrics = evaluate_structures(zip_file, chgnet_model=chgnet)
        
        # Add parameters to metrics
        metrics.update(params)
        
        return metrics
        
    except Exception as e:
        print(f"Error in trial {trial_id}: {e}")
        metrics = {
            'error': str(e),
            'status': 'failed'
        }
        metrics.update(params)
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Tune magnetic density generation")
    parser.add_argument("--results-dir", type=str, default="tuning_results",
                      help="Directory to store results")
    parser.add_argument("--num-trials", type=int, default=50,
                      help="Number of random trials from parameter grid")
    parser.add_argument("--batch-size", type=int, default=4,
                      help="Batch size for generation")
    parser.add_argument("--num-batches", type=int, default=1,
                      help="Number of batches to generate")
    args = parser.parse_args()
    
    # Parameter grid
    param_grid = {
        'guidance_scale': [1.0, 2.0, 5.0, 10.0, 20.0],
        'density_weight': [1.0, 2.0, 5.0, 10.0],
        'moment_weight': [0.0, 0.3, 0.6, 1.0],
        'volume_weight': [0.0, 0.2, 0.4],
        'target_volume': [150.0, 200.0, 250.0]
    }
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    
    # Randomly sample combinations if num_trials is less than total combinations
    num_combinations = len(all_combinations)
    if args.num_trials < num_combinations:
        selected_combinations = [all_combinations[i] for i in 
                               np.random.choice(num_combinations, 
                                              args.num_trials, 
                                              replace=False)]
    else:
        selected_combinations = all_combinations
    
    # Run trials
    results = []
    for trial_id, params in enumerate(selected_combinations):
        param_dict = dict(zip(param_names, params))
        print(f"\nRunning trial {trial_id} with parameters:")
        for k, v in param_dict.items():
            print(f"  {k}: {v}")
        
        metrics = run_trial(
            trial_id=trial_id,
            results_dir=results_dir,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            **param_dict
        )
        results.append(metrics)
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(results_dir / "results.csv", index=False)
        
        print(f"\nTrial {trial_id} results:")
        print(f"  Average density: {metrics['avg_density']:.6f} μB/Å³")
        print(f"  Max density: {metrics['max_density']:.6f} μB/Å³")
        print(f"  % above 0.1 target: {metrics['pct_above_0.1_target']*100:.1f}%")
        print(f"  % above 0.5 target: {metrics['pct_above_0.5_target']*100:.1f}%")
    
    print("\nTuning completed. Results saved to:", results_dir)

if __name__ == "__main__":
    main() 