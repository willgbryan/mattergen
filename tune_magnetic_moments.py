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
import random

def compute_alignment(magnetic_moments: np.ndarray) -> float:
    """
    Compute the alignment score for magnetic moments.
    
    The alignment score is 1.0 if all moments are perfectly aligned (same sign),
    0.0 if randomly aligned, and -1.0 if perfectly anti-aligned.
    
    Args:
        magnetic_moments: Array of magnetic moments for each atom
        
    Returns:
        Alignment score between -1.0 and 1.0
    """
    # Convert to tensor if numpy array
    if isinstance(magnetic_moments, np.ndarray):
        magnetic_moments = torch.tensor(magnetic_moments)
    
    # Get signs of moments, ignoring near-zero values
    signs = torch.sign(magnetic_moments)
    mask = torch.abs(magnetic_moments) > 1e-6  # Ignore near-zero moments
    
    if not torch.any(mask):
        return 0.0  # All moments are effectively zero
    
    # Calculate mean sign, considering only non-zero moments
    mean_sign = torch.abs(signs[mask].float().mean())
    
    # Enhanced logging for alignment calculation
    print("\nAlignment Details:")
    print(f"Raw moments: {magnetic_moments.tolist()}")
    print(f"Signs: {signs.tolist()}")
    print(f"Significant moments mask: {mask.tolist()}")
    print(f"Number of significant moments: {torch.sum(mask).item()}")
    print(f"Mean sign (absolute): {mean_sign:.4f}")
    
    # Additional validation checks
    if mean_sign > 0.99:  # Near perfect alignment
        print("WARNING: Perfect alignment detected - validating...")
        significant_moments = magnetic_moments[mask]
        max_moment = torch.max(torch.abs(significant_moments))
        relative_moments = significant_moments / max_moment
        min_alignment = torch.min(torch.abs(relative_moments))
        print(f"Minimum relative alignment: {min_alignment:.4f}")
        if min_alignment < 0.01:
            print("WARNING: Possible false positive - some moments nearly zero")
    
    return float(mean_sign)

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
        
        # Ultra-aggressive thresholds and multipliers
        self.thresholds = [
            (5.0, 3.0),     # 3x bonus above 5 μB
            (10.0, 5.0),    # 5x bonus above 10 μB
            (15.0, 8.0),    # 8x bonus above 15 μB
            (20.0, 12.0),   # 12x bonus above 20 μB
            (25.0, 16.0),   # 16x bonus above 25 μB
            (30.0, 20.0)    # 20x bonus above 30 μB
        ]
    
    def compute_reward(self, batch: dict) -> torch.Tensor:
        structures = self._batch_to_structures(batch)
        rewards = []
        
        for structure in structures:
            prediction = self.chgnet_model.predict_structure(structure)
            mag_moments = prediction['m']
            mag_moment = float(sum(mag_moments))
            num_atoms = len(structure)
            
            # More aggressive base reward with seventh power scaling
            base_reward = (mag_moment ** 7) / (num_atoms ** 6)
            
            # Progressive threshold bonuses
            for threshold, multiplier in self.thresholds:
                if mag_moment > threshold:
                    base_reward *= multiplier
            
            # Enhanced alignment bonus with proper alignment calculation
            alignment = compute_alignment(mag_moments)
            if alignment > 0.9:  # Stricter alignment requirement
                base_reward *= 4.0  # Higher alignment bonus
            elif alignment > 0.8:
                base_reward *= 2.0
            
            # Enhanced per-atom moment bonuses
            per_atom_moment = mag_moment / num_atoms
            if per_atom_moment > 2.0:
                base_reward *= 2.0
            if per_atom_moment > 3.0:
                base_reward *= 3.0
            if per_atom_moment > 4.0:  # New higher threshold
                base_reward *= 4.0
            
            rewards.append(base_reward)
        
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
        'per_atom_moments': [],
        'volumes': [],
        'alignments': [],
        'num_atoms': [],
        'compositions': [],  # Track atomic compositions
        'atomic_moments': [],  # Track per-atom moment distributions
        'alignment_details': []  # New field for detailed alignment info
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
                
                # Enhanced alignment analysis
                alignment_score = compute_alignment(mag_moments)
                alignment_detail = {
                    'score': alignment_score,
                    'raw_moments': mag_moments.tolist(),
                    'significant_moments': (torch.abs(torch.tensor(mag_moments)) > 1e-6).tolist(),
                    'composition': structure.composition.reduced_formula
                }
                
                metrics['moments'].append(mag_moment)
                metrics['per_atom_moments'].append(mag_moment / num_atoms)
                metrics['volumes'].append(volume)
                metrics['alignments'].append(alignment_score)
                metrics['num_atoms'].append(num_atoms)
                metrics['compositions'].append(structure.composition.as_dict())
                metrics['atomic_moments'].append(mag_moments.tolist())
                metrics['alignment_details'].append(alignment_detail)
                
                # Log detailed analysis for high-moment structures
                if mag_moment > 10.0:
                    print(f"\nHigh-moment structure found ({mag_moment:.2f} μB):")
                    print(f"Composition: {structure.composition.reduced_formula}")
                    print(f"Alignment score: {alignment_score:.4f}")
                    print("Per-atom moments:")
                    for site, moment in zip(structure.sites, mag_moments):
                        print(f"{site.specie}: {moment:.4f} μB")
                
            except Exception as e:
                print(f"Error evaluating structure from {f}: {e}")
    
    if not metrics['moments']:
        raise ValueError("No valid structures were evaluated")
    
    # Convert to numpy arrays where appropriate
    for key in ['moments', 'per_atom_moments', 'volumes', 'alignments', 'num_atoms']:
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
    
    # Add composition analysis for highest moment structure
    max_moment_idx = np.argmax(metrics['moments'])
    summary['best_structure_composition'] = metrics['compositions'][max_moment_idx]
    summary['best_structure_atomic_moments'] = metrics['atomic_moments'][max_moment_idx]
    
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
    parser.add_argument("--results-dir", type=str, default="moment_tuning_results_v8",
                      help="Directory to store results")
    parser.add_argument("--num-trials", type=int, default=20,
                      help="Number of random trials from parameter grid")
    parser.add_argument("--batch-size", type=int, default=16,
                      help="Batch size for generation")
    parser.add_argument("--num-batches", type=int, default=3,
                      help="Number of batches to generate")
    args = parser.parse_args()
    
    # Even more aggressive parameter grid
    param_grid = {
        'guidance_scale': [20000.0, 25000.0, 30000.0],  # Higher guidance scales
        'moment_weight': [2500.0, 3000.0, 3500.0],      # Higher moment weights
    }
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameter grid for reference
    with open(results_dir / "param_grid.json", "w") as f:
        json.dump(param_grid, f, indent=2)
    
    print("\nStarting magnetic moment optimization with parameters:")
    print(f"Guidance scales: {param_grid['guidance_scale']}")
    print(f"Moment weights: {param_grid['moment_weight']}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per trial: {args.num_batches}")
    
    # Run trials
    results = []
    for trial_id in range(args.num_trials):
        print(f"\nTrial {trial_id + 1}/{args.num_trials}")
        
        # Randomly select parameters from grid
        params = {
            'guidance_scale': random.choice(param_grid['guidance_scale']),
            'moment_weight': random.choice(param_grid['moment_weight'])
        }
        
        metrics = run_trial(
            trial_id=trial_id,
            results_dir=results_dir,
            guidance_scale=params['guidance_scale'],
            moment_weight=params['moment_weight'],
            batch_size=args.batch_size,
            num_batches=args.num_batches
        )
        results.append(metrics)
        
        # Save results after each trial
        df = pd.DataFrame(results)
        df.to_csv(results_dir / "results.csv", index=False)
        
        if 'magnetic' in metrics:
            print(f"\nTrial {trial_id + 1} results:")
            print(f"Average moment: {metrics['magnetic']['moment']['mean']:.2f} μB")
            print(f"Max moment: {metrics['magnetic']['moment']['max']:.2f} μB")
            if 'best_structure' in metrics:
                print(f"Best structure composition: {metrics['best_structure']['composition']}")
                print(f"Best structure moment: {metrics['best_structure']['moment']:.2f} μB")
    
    print("\nOptimization complete. Results saved to:", results_dir)

if __name__ == "__main__":
    main() 