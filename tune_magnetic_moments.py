"""
Magnetic moment maximization tuning script.

This script focuses solely on maximizing magnetic moments, without considering density.
The goal is to understand our baseline control over magnetic properties.
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

class MomentMaximizationReward(BaseRewardFunction):
    """Simple reward function focused solely on maximizing magnetic moments through atom selection."""
    
    def __init__(
        self,
        moment_weight: float = 1.0,
        chgnet_model=None
    ):
        """
        Args:
            moment_weight: Weight for moment maximization term
            chgnet_model: Pre-loaded CHGNet model
        """
        from chgnet.model.model import CHGNet
        self.moment_weight = moment_weight
        self.chgnet_model = chgnet_model or CHGNet.load()
        
        # For logging
        self.current_rewards = []
        self.current_moments = []
        self.current_per_atom_moments = []
    
    def compute_reward(self, batch: dict) -> torch.Tensor:
        structures = self._batch_to_structures(batch)
        atomic_rewards = []  # Rewards for atom type selection
        
        # Clear previous batch metrics
        self.current_rewards = []
        self.current_moments = []
        self.current_per_atom_moments = []
        
        print("\nComputing magnetic moment rewards for batch:")
        for i, structure in enumerate(structures):
            prediction = self.chgnet_model.predict_structure(structure)
            mag_moments = prediction['m']  # Per-atom array
            mag_moment = float(sum(mag_moments))  # Total magnetic moment
            num_atoms = len(structure)
            per_atom_moment = mag_moment / num_atoms
            
            # Simple quadratic reward that grows faster with larger moments
            # Normalized by number of atoms to make it comparable across structures
            base_reward = (mag_moment ** 2) / num_atoms
            atomic_reward = self.moment_weight * base_reward
            
            # Store metrics for logging
            self.current_rewards.append(float(atomic_reward))
            self.current_moments.append(mag_moment)
            self.current_per_atom_moments.append(per_atom_moment)
            
            print(f"  Structure {i}:")
            print(f"    Number of atoms: {num_atoms}")
            print(f"    Total magnetic moment: {mag_moment:.3f} μB")
            print(f"    Per-atom moment: {per_atom_moment:.3f} μB/atom")
            print(f"    Atomic reward: {atomic_reward:.3f}")
            
            atomic_rewards.append(atomic_reward)
        
        # Only return rewards for atomic_numbers
        rewards = {
            'atomic_numbers': torch.stack(atomic_rewards)
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
        'per_atom_moments': [],
        'volumes': [],
        'alignments': [],
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
                
                mag_moments = prediction['m']
                mag_moment = float(sum(mag_moments))
                volume = structure.volume
                num_atoms = len(structure)
                
                # Calculate moment alignment
                moment_signs = torch.sign(torch.tensor(mag_moments))
                alignment_score = float(torch.abs(moment_signs.float().mean()))
                
                metrics['moments'].append(mag_moment)
                metrics['per_atom_moments'].append(mag_moment / num_atoms)
                metrics['volumes'].append(volume)
                metrics['alignments'].append(alignment_score)
                metrics['num_atoms'].append(num_atoms)
                
            except Exception as e:
                print(f"Error evaluating structure from {f}: {e}")
    
    if not metrics['moments']:
        raise ValueError("No valid structures were evaluated")
    
    # Convert to numpy arrays
    for key in metrics:
        metrics[key] = np.array(metrics[key])
    
    # Compute summary statistics
    summary = {
        'avg_moment': float(np.mean(metrics['moments'])),
        'std_moment': float(np.std(metrics['moments'])),
        'max_moment': float(np.max(metrics['moments'])),
        'min_moment': float(np.min(metrics['moments'])),
        'avg_per_atom_moment': float(np.mean(metrics['per_atom_moments'])),
        'max_per_atom_moment': float(np.max(metrics['per_atom_moments'])),
        'avg_alignment': float(np.mean(metrics['alignments'])),
        'avg_volume': float(np.mean(metrics['volumes'])),
        'avg_num_atoms': float(np.mean(metrics['num_atoms'])),
        'pct_above_10_moment': float(np.mean(metrics['moments'] > 10.0)),
        'pct_above_20_moment': float(np.mean(metrics['moments'] > 20.0)),
        'num_structures': len(metrics['moments'])
    }
    
    return summary

def run_trial(
    trial_id: int,
    results_dir: Path,
    guidance_scale: float,
    moment_weight: float,
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
        'moment_weight': moment_weight,
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
        reward_fn = MomentMaximizationReward(
            moment_weight=moment_weight,
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
    parser = argparse.ArgumentParser(description="Tune magnetic moment maximization")
    parser.add_argument("--results-dir", type=str, default="moment_tuning_results_v3",
                      help="Directory to store results")
    parser.add_argument("--num-trials", type=int, default=20,
                      help="Number of random trials from parameter grid")
    parser.add_argument("--batch-size", type=int, default=4,
                      help="Batch size for generation")
    parser.add_argument("--num-batches", type=int, default=1,
                      help="Number of batches to generate")
    args = parser.parse_args()
    
    # Parameter grid - focused on atom selection for magnetic properties
    param_grid = {
        'guidance_scale': [500.0, 1000.0, 2000.0],     # Lower scales since we're only guiding atoms
        'moment_weight': [100.0, 200.0, 500.0],        # Wider range to explore atom selection impact
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
        print(f"\nRunning trial {trial_id}")
        print(f"Parameters: {dict(zip(param_names, params))}")
        
        metrics = run_trial(
            trial_id=trial_id,
            results_dir=results_dir,
            guidance_scale=params[0],
            moment_weight=params[1],
            batch_size=args.batch_size,
            num_batches=args.num_batches
        )
        results.append(metrics)
    
    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "results.csv", index=False)
    print("\nAll trials complete. Results saved to:", results_dir / "results.csv")

if __name__ == "__main__":
    main() 