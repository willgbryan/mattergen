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
        pos_rewards = []  # Rewards for atomic positions
        cell_rewards = [] # Rewards for cell parameters
        
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
            num_atoms = len(structure)
            
            # 1. Position-specific rewards (focus on maximizing moments)
            # Calculate target moment needed for this volume
            target_moment = self.target_magnetic_density * volume
            moment_ratio = mag_moment / target_moment
            
            # Simple linear reward for atomic moments
            # Reward grows linearly with moment ratio, no early penalties
            pos_reward = self.moment_weight * (moment_ratio - 0.1)
            
            # Add small bonus for aligned moments
            moment_signs = torch.sign(torch.tensor(mag_moments))
            alignment_score = torch.abs(moment_signs.float().mean())
            pos_reward += 0.2 * self.moment_weight * (alignment_score - 0.5)
            
            # 2. Cell-specific rewards (focus on density directly)
            # Calculate optimal volume range based on number of atoms
            min_vol = num_atoms * 8.0   # Minimum 8 Å³ per atom
            max_vol = num_atoms * 20.0  # Maximum 20 Å³ per atom
            
            # Volume reward: quadratic within bounds, linear outside
            if volume < min_vol:
                vol_ratio = volume / min_vol
                volume_reward = -self.volume_weight * (1.0 - vol_ratio)
            elif volume > max_vol:
                vol_ratio = volume / max_vol
                volume_reward = -0.5 * self.volume_weight * (vol_ratio - 1.0)
            else:
                # Small positive reward for good volume range
                volume_reward = 0.1 * self.volume_weight
            
            # Direct density reward - linear with bonus thresholds
            density_ratio = mag_density / self.target_magnetic_density
            density_reward = self.density_weight * density_ratio
            
            # Add bonuses for crossing thresholds
            if density_ratio > 0.5:
                density_reward *= 2.0  # Double reward above 50%
            elif density_ratio > 0.2:
                density_reward *= 1.5  # 50% bonus above 20%
            elif density_ratio > 0.1:
                density_reward *= 1.2  # 20% bonus above 10%
            
            # Combine cell rewards
            cell_reward = density_reward + volume_reward
            
            # Store metrics for logging
            self.current_rewards.append(float(pos_reward + cell_reward))
            self.current_densities.append(mag_density)
            self.current_moments.append(mag_moment)
            self.current_volumes.append(volume)
            
            print(f"  Structure {i}:")
            print(f"    Volume: {volume:.3f} Å³")
            print(f"    Number of atoms: {num_atoms}")
            print(f"    Total magnetic moment: {mag_moment:.3f} μB")
            print(f"    Target moment needed: {target_moment:.3f} μB")
            print(f"    Moment ratio: {moment_ratio:.3f}")
            print(f"    Magnetic density: {mag_density:.6f} μB/Å³")
            print(f"    Target density: {self.target_magnetic_density:.6f} μB/Å³")
            print(f"    Density ratio: {density_ratio:.3f}")
            print(f"    Position rewards:")
            print(f"      Moment reward: {pos_reward:.3f}")
            print(f"    Cell rewards:")
            print(f"      Volume reward: {volume_reward:.3f}")
            print(f"      Density reward: {density_reward:.3f}")
            print(f"      Total cell reward: {cell_reward:.3f}")
            
            pos_rewards.append(pos_reward)
            cell_rewards.append(cell_reward)
        
        # Return dictionary of rewards for different components
        rewards = {
            'pos': torch.stack(pos_rewards),
            'cell': torch.stack(cell_rewards)
        }
        return rewards

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
        'guidance_scale': [500.0, 1000.0, 2000.0],      # Much stronger guidance
        'density_weight': [10.0, 20.0, 50.0],           # Direct density emphasis
        'moment_weight': [50.0, 100.0, 200.0],          # Very strong moment emphasis
        'volume_weight': [0.1],                         # Minimal volume control
        'target_volume': [100.0]                        # Smaller target volume
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