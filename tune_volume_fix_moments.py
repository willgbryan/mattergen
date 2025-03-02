"""
Magnetic moment optimization with volume control.

This script extends our successful magnetic moment optimization with volume control
to achieve higher magnetic densities. Key features:
1. Maintains successful moment optimization parameters
2. Adds volume guidance through cell predictor
3. Progressive rewards for magnetic density achievements
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
import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

def compute_alignment(magnetic_moments: np.ndarray) -> float:
    """Compute alignment score for magnetic moments (from tune_magnetic_moments.py)."""
    if isinstance(magnetic_moments, np.ndarray):
        magnetic_moments = torch.tensor(magnetic_moments)
    
    signs = torch.sign(magnetic_moments)
    mask = torch.abs(magnetic_moments) > 1e-6
    
    if not torch.any(mask):
        return 0.0
    
    mean_sign = torch.abs(signs[mask].float().mean())
    return float(mean_sign)

class DensityOptimizedMomentReward(BaseRewardFunction):
    """Reward function optimizing magnetic moments with volume control and progressive density rewards."""
    
    def __init__(
        self,
        moment_weight: float = 1.0,
        volume_weight: float = 1.0,
        target_density: float = 0.9,    # Target in μB/Å³ (based on NdFeB magnets)
        chgnet_model=None
    ):
        """
        Args:
            moment_weight: Weight for magnetic moment component
            volume_weight: Weight for volume control component
            target_density: Target magnetic density in μB/Å³
            chgnet_model: Pre-loaded CHGNet model, will load default if None
        """
        from chgnet.model.model import CHGNet
        self.moment_weight = moment_weight
        self.volume_weight = volume_weight
        self.target_density = target_density
        self.chgnet_model = chgnet_model or CHGNet.load()
        
        # Volume constraints with wider range for exploration
        self.min_volume = 50.0  # Å³
        self.max_volume = 400.0  # Å³
        
        # Progressive moment thresholds with smaller steps
        self.moment_thresholds = [
            (2.0, 1.2),    # Small bonus to encourage initial magnetism
            (5.0, 1.5),    # Moderate bonus for basic magnetic ordering
            (10.0, 2.0),   # Good bonus for strong magnetism
            (15.0, 3.0),   # Strong bonus for very strong magnetism
            (20.0, 4.0),   # Major bonus for exceptional magnetism
            (25.0, 5.0)    # Maximum bonus for outstanding magnetism
        ]
        
        # Progressive density thresholds
        self.density_thresholds = [
            (0.05, 1.2),   # Small bonus to encourage initial density progress
            (0.1, 1.5),    # Better bonus as density improves
            (0.15, 2.0),   # Good bonus approaching our current best
            (0.2, 3.0),    # Strong bonus for exceeding current best
            (0.3, 4.0),    # Major bonus for significant improvement
            (0.5, 5.0)     # Outstanding bonus for approaching target
        ]
        
        # Track best structure
        self.best_density = 0.0
        self.best_moment = 0.0
        self.best_volume = 0.0
        self.best_reward = float('-inf')
    
    def compute_reward(self, batch: dict) -> torch.Tensor:
        """Compute reward based on magnetic moments with progressive density rewards."""
        structures = self._batch_to_structures(batch)
        rewards = []
        
        for structure in structures:
            try:
                prediction = self.chgnet_model.predict_structure(structure)
                mag_moments = prediction['m']
                mag_moment = float(sum(mag_moments))
                volume = structure.volume
                mag_density = mag_moment / volume
                
                # 1. Base moment reward with progressive thresholds
                moment_multiplier = 1.0
                for threshold, bonus in self.moment_thresholds:
                    if mag_moment >= threshold:
                        moment_multiplier = bonus
                moment_reward = mag_moment * moment_multiplier
                
                # 2. Alignment bonus (1.0 to 2.0 multiplier)
                alignment = compute_alignment(mag_moments)
                moment_reward *= (1.0 + alignment)
                
                # 3. Density bonus with progressive thresholds
                density_multiplier = 1.0
                for threshold, bonus in self.density_thresholds:
                    if mag_density >= threshold:
                        density_multiplier = bonus
                moment_reward *= density_multiplier
                
                # 4. Softer volume control
                if volume < self.min_volume or volume > self.max_volume:
                    # Soft boundary penalty
                    if volume < self.min_volume:
                        volume_penalty = -2.0 * (self.min_volume - volume) / self.min_volume
                    else:
                        volume_penalty = -1.0 * (volume - self.max_volume) / self.max_volume
                else:
                    # Guide towards ideal volume based on current moment
                    ideal_volume = mag_moment / self.target_density if mag_moment > 0 else 100.0
                    ideal_volume = max(self.min_volume, min(self.max_volume, ideal_volume))
                    
                    # Softer penalty for volume deviation
                    volume_diff = abs(volume - ideal_volume) / ideal_volume
                    volume_penalty = -0.5 * volume_diff  # Reduced penalty factor
                
                # Combine rewards with weights
                reward = (self.moment_weight * moment_reward + 
                         self.volume_weight * volume_penalty)
                
                # Track best structure
                if mag_density > self.best_density:
                    self.best_density = mag_density
                    self.best_moment = mag_moment
                    self.best_volume = volume
                    self.best_reward = float(reward)
                
                # Log high-density structures with progressive thresholds
                if mag_density > 0.05:  # Lower threshold for logging
                    print(f"\nDensity milestone reached!")
                    print(f"Total moment: {mag_moment:.2f} μB")
                    print(f"Volume: {volume:.2f} Å³")
                    print(f"Density: {mag_density:.4f} μB/Å³")
                    print(f"Alignment score: {alignment:.2f}")
                    print(f"Reward components:")
                    print(f"  Moment reward: {moment_reward:.2f}")
                    print(f"  Volume penalty: {volume_penalty:.2f}")
                    print(f"  Moment multiplier: {moment_multiplier:.1f}x")
                    print(f"  Density multiplier: {density_multiplier:.1f}x")
                    print(f"  Total reward: {reward:.2f}")
                
                rewards.append(reward)
            
            except Exception as e:
                print(f"Error computing reward: {e}")
                rewards.append(torch.tensor(-1000.0))
        
        return torch.tensor(rewards, device=batch['pos'].device)

def evaluate_structures(zip_file: Path, chgnet_model=None) -> dict:
    """Enhanced evaluation function with density metrics."""
    if not zip_file.exists():
        raise FileNotFoundError(f"No zip file found at {zip_file}")
    
    if chgnet_model is None:
        chgnet_model = CHGNet.load()
    
    metrics = {
        'moments': [],
        'volumes': [],
        'densities': [],
        'alignments': [],
        'compositions': [],
        'atomic_moments': []
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
                
                mag_moments = prediction['m']
                mag_moment = float(sum(mag_moments))
                volume = structure.volume
                density = mag_moment / volume
                alignment = compute_alignment(mag_moments)
                
                metrics['moments'].append(mag_moment)
                metrics['volumes'].append(volume)
                metrics['densities'].append(density)
                metrics['alignments'].append(alignment)
                metrics['compositions'].append(structure.composition.as_dict())
                metrics['atomic_moments'].append(mag_moments.tolist())
                
                # Log high-density structures
                if density > 0.2:
                    print(f"\nHigh-density structure in {f.name}:")
                    print(f"Composition: {structure.composition.reduced_formula}")
                    print(f"Magnetic moment: {mag_moment:.2f} μB")
                    print(f"Volume: {volume:.2f} Å³")
                    print(f"Density: {density:.4f} μB/Å³")
                    print(f"Alignment: {alignment:.4f}")
                
            except Exception as e:
                print(f"Error evaluating structure from {f}: {e}")
    
    if not metrics['moments']:
        raise ValueError("No valid structures were evaluated")
    
    # Convert to numpy arrays
    for key in ['moments', 'volumes', 'densities', 'alignments']:
        metrics[key] = np.array(metrics[key])
    
    # Compute summary statistics
    summary = {
        'magnetic': {
            'moment': {
                'mean': float(np.mean(metrics['moments'])),
                'std': float(np.std(metrics['moments'])),
                'max': float(np.max(metrics['moments'])),
                'min': float(np.min(metrics['moments']))
            },
            'density': {
                'mean': float(np.mean(metrics['densities'])),
                'std': float(np.std(metrics['densities'])),
                'max': float(np.max(metrics['densities'])),
                'min': float(np.min(metrics['densities'])),
                'pct_above_0.2': float(np.mean(metrics['densities'] > 0.2)),
                'pct_above_0.5': float(np.mean(metrics['densities'] > 0.5))
            }
        },
        'volume': {
            'mean': float(np.mean(metrics['volumes'])),
            'std': float(np.std(metrics['volumes'])),
            'min': float(np.min(metrics['volumes'])),
            'max': float(np.max(metrics['volumes']))
        },
        'alignment': {
            'mean': float(np.mean(metrics['alignments'])),
            'pct_above_0.8': float(np.mean(metrics['alignments'] > 0.8))
        }
    }
    
    # Add best structure details
    best_density_idx = np.argmax(metrics['densities'])
    summary['best_structure'] = {
        'composition': metrics['compositions'][best_density_idx],
        'moment': float(metrics['moments'][best_density_idx]),
        'volume': float(metrics['volumes'][best_density_idx]),
        'density': float(metrics['densities'][best_density_idx]),
        'alignment': float(metrics['alignments'][best_density_idx])
    }
    
    return summary

def run_trial(
    trial_id: int,
    results_dir: Path,
    guidance_scale: float,
    moment_weight: float,
    volume_weight: float,
    batch_size: int = 16,
    num_batches: int = 3
) -> dict:
    """Run a single trial with the specified parameters."""
    # Create trial directory
    trial_dir = results_dir / f'trial_{trial_id}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nStarting trial {trial_id}")
    print(f"Results will be saved to: {trial_dir}")
    
    # Initialize reward function with trial directory
    reward_function = DensityOptimizedMomentReward(
        moment_weight=moment_weight,
        volume_weight=volume_weight
    )
    
    # Create generator
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
        batch_size=batch_size,
        num_batches=num_batches
    )
    
    # Add property guidance
    guided_generator = create_property_guided_generator(
        generator=generator,
        reward_function=reward_function,
        guidance_scale=guidance_scale
    )
    
    # Generate structures
    print(f"\nGenerating structures with parameters:")
    print(f"  Guidance scale: {guidance_scale}")
    print(f"  Moment weight: {moment_weight}")
    print(f"  Volume weight: {volume_weight}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    
    guided_generator.generate(output_dir=trial_dir)
    
    # Evaluate results
    zip_file = trial_dir / "generated_crystals_cif.zip"
    if not zip_file.exists():
        print(f"Warning: No structures generated in trial {trial_id}")
        return {}
    
    print(f"\nEvaluating structures from trial {trial_id}")
    metrics = evaluate_structures(zip_file)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Tune magnetic density optimization")
    parser.add_argument("--results-dir", type=str, default="density_tuning_results",
                      help="Directory to store results")
    parser.add_argument("--num-trials", type=int, default=20,
                      help="Number of random trials")
    parser.add_argument("--batch-size", type=int, default=16,
                      help="Batch size for generation")
    parser.add_argument("--num-batches", type=int, default=3,
                      help="Number of batches to generate")
    args = parser.parse_args()
    
    # Parameter grid with successful high values
    param_grid = {
        'guidance_scale': [20000.0, 25000.0, 30000.0],  # Higher guidance scales
        'moment_weight': [2500.0, 3000.0, 3500.0],      # Higher moment weights
        'volume_weight': [0.5, 1.0, 2.0]                # New volume control
    }
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated results directory: {results_dir}")
    
    print("\nStarting magnetic density optimization with high-value parameters:")
    print(f"Results directory: {results_dir}")
    print(f"Guidance scales: {param_grid['guidance_scale']}")
    print(f"Moment weights: {param_grid['moment_weight']}")
    print(f"Volume weights: {param_grid['volume_weight']}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per trial: {args.num_batches}")
    print(f"Number of trials: {args.num_trials}")
    
    # Save parameter grid
    param_grid_file = results_dir / "param_grid.json"
    with open(param_grid_file, "w") as f:
        json.dump(param_grid, f, indent=2)
    print(f"Saved parameter grid to: {param_grid_file}")
    
    results = []
    results_file = results_dir / "results.csv"
    
    for trial_id in range(args.num_trials):
        print(f"\n{'='*50}")
        print(f"Starting Trial {trial_id + 1}/{args.num_trials}")
        print(f"{'='*50}")
        
        # Create trial directory
        trial_dir = results_dir / f"trial_{trial_id}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created trial directory: {trial_dir}")
        
        # Randomly sample parameters
        params = {
            'guidance_scale': random.choice(param_grid['guidance_scale']),
            'moment_weight': random.choice(param_grid['moment_weight']),
            'volume_weight': random.choice(param_grid['volume_weight'])
        }
        
        print("\nSelected parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        try:
            metrics = run_trial(
                trial_id=trial_id,
                results_dir=trial_dir,
                guidance_scale=params['guidance_scale'],
                moment_weight=params['moment_weight'],
                volume_weight=params['volume_weight'],
                batch_size=args.batch_size,
                num_batches=args.num_batches
            )
            metrics.update(params)
            results.append(metrics)
            
            # Save results after each trial
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_file, index=False)
            print(f"\nUpdated results saved to: {results_file}")
            
            if 'magnetic' in metrics:
                print(f"\nTrial {trial_id + 1} summary:")
                print(f"Average moment: {metrics['magnetic']['moment']['mean']:.2f} μB")
                print(f"Max moment: {metrics['magnetic']['moment']['max']:.2f} μB")
                print(f"Average density: {metrics['magnetic']['density']['mean']:.4f} μB/Å³")
                print(f"Max density: {metrics['magnetic']['density']['max']:.4f} μB/Å³")
                print(f"% above 0.2 μB/Å³: {metrics['magnetic']['density']['pct_above_0.2']*100:.1f}%")
            
        except Exception as e:
            print(f"Error in trial {trial_id}:")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {str(e)}")
            if hasattr(e, '__traceback__'):
                import traceback
                print("  Traceback:")
                traceback.print_tb(e.__traceback__)
            continue
    
    print("\nOptimization complete!")
    print(f"Final results saved to: {results_file}")

if __name__ == "__main__":
    main() 