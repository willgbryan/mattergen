"""
Test script for physics-aware magnetic guidance.

This script evaluates our enhanced magnetic guidance system that incorporates
physical principles directly into the diffusion process.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import zipfile
import tempfile
from chgnet.model.model import CHGNet
from mattergen.generator import CrystalGenerator
from mattergen.diffusion.sampling.physics_aware_guidance import PhysicsAwareMagneticGuidance
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from pymatgen.core import Structure

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
        'coupling_scores': [],
        'num_atoms': [],
        'compositions': [],
        'atomic_moments': [],
        'nearest_neighbor_dists': []
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
                num_atoms = len(structure)
                
                # Compute coupling score based on magnetic site distances
                magnetic_sites = [i for i, m in enumerate(mag_moments) if abs(m) > 0.1]
                coupling_score = 0.0
                nn_dists = []
                
                if len(magnetic_sites) >= 2:
                    for i in magnetic_sites:
                        neighbors = structure.get_neighbors(structure[i], r=4.0)
                        for neighbor, dist in neighbors:
                            j = neighbor.index
                            if j in magnetic_sites:
                                # Score based on optimal coupling distance (2.5 Å)
                                coupling_score += np.exp(-(dist - 2.5)**2 / (2 * 0.5**2))
                                nn_dists.append(dist)
                
                metrics['moments'].append(mag_moment)
                metrics['densities'].append(density)
                metrics['volumes'].append(volume)
                metrics['coupling_scores'].append(coupling_score)
                metrics['num_atoms'].append(num_atoms)
                metrics['compositions'].append(structure.composition.as_dict())
                metrics['atomic_moments'].append(mag_moments.tolist())
                metrics['nearest_neighbor_dists'].extend(nn_dists)
                
                # Log detailed analysis for high-density structures
                if density > 0.5:  # μB/Å³
                    print(f"\nHigh-density structure found ({density:.3f} μB/Å³):")
                    print(f"Composition: {structure.composition.reduced_formula}")
                    print(f"Total moment: {mag_moment:.2f} μB")
                    print(f"Volume: {volume:.2f} Å³")
                    print(f"Coupling score: {coupling_score:.2f}")
                    print("Per-atom moments:")
                    for site, moment in zip(structure.sites, mag_moments):
                        print(f"{site.specie}: {moment:.4f} μB")
                
            except Exception as e:
                print(f"Error evaluating structure from {f}: {e}")
    
    if not metrics['moments']:
        raise ValueError("No valid structures were evaluated")
    
    # Convert to numpy arrays
    for key in ['moments', 'densities', 'volumes', 'coupling_scores', 'num_atoms']:
        metrics[key] = np.array(metrics[key])
    
    # Compute summary statistics
    summary = {
        'avg_moment': float(np.mean(metrics['moments'])),
        'max_moment': float(np.max(metrics['moments'])),
        'avg_density': float(np.mean(metrics['densities'])),
        'max_density': float(np.max(metrics['densities'])),
        'avg_coupling_score': float(np.mean(metrics['coupling_scores'])),
        'avg_volume': float(np.mean(metrics['volumes'])),
        'avg_num_atoms': float(np.mean(metrics['num_atoms'])),
        'avg_nn_dist': float(np.mean(metrics['nearest_neighbor_dists'])),
        'pct_above_0.1_density': float(np.mean(metrics['densities'] > 0.1)),
        'pct_above_0.5_density': float(np.mean(metrics['densities'] > 0.5)),
        'num_structures': len(metrics['moments'])
    }
    
    # Add composition analysis for highest density structure
    max_density_idx = np.argmax(metrics['densities'])
    summary['best_structure_composition'] = metrics['compositions'][max_density_idx]
    summary['best_structure_atomic_moments'] = metrics['atomic_moments'][max_density_idx]
    
    return summary

def run_trial(
    trial_id: int,
    results_dir: Path,
    guidance_scale: float,
    coupling_weight: float,
    alignment_weight: float,
    target_density: float = 0.9,
    batch_size: int = 4,
    num_batches: int = 1
) -> dict:
    """Run a single trial with specified hyperparameters."""
    
    trial_dir = results_dir / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting trial {trial_id}")
    print(f"Parameters:")
    print(f"- Guidance scale: {guidance_scale}")
    print(f"- Coupling weight: {coupling_weight}")
    print(f"- Alignment weight: {alignment_weight}")
    print(f"- Target density: {target_density}")
    
    # Load CHGNet model (shared between reward and evaluation)
    chgnet = CHGNet.load()
    
    # Create base generator
    generator = CrystalGenerator(
        checkpoint_info=MatterGenCheckpointInfo.from_hf_hub("mattergen_base"),
        batch_size=batch_size,
        num_batches=num_batches,
        record_trajectories=True
    )
    
    # Create physics-aware guided generator
    guided_generator = generator
    guided_generator._model.sampler = PhysicsAwareMagneticGuidance(
        guidance_scale=guidance_scale,
        target_density=target_density,
        coupling_weight=coupling_weight,
        alignment_weight=alignment_weight,
        chgnet_model=chgnet,
        remove_conditioning_fn=guided_generator._model.sampler.remove_conditioning_fn,
        keep_conditioning_fn=guided_generator._model.sampler.keep_conditioning_fn,
        **{k: v for k, v in guided_generator._model.sampler.__dict__.items() 
           if k not in ['guidance_scale', 'remove_conditioning_fn', 'keep_conditioning_fn']}
    )
    
    # Generate structures
    try:
        guided_generator.generate(output_dir=trial_dir)
        
        # Get statistics from guidance
        guidance_stats = guided_generator._model.sampler.get_statistics()
        
        # Evaluate generated structures
        zip_file = trial_dir / "generated_crystals_cif.zip"
        eval_results = evaluate_structures(zip_file, chgnet_model=chgnet)
        
        # Combine results
        results = {
            'trial_id': trial_id,
            'guidance_scale': guidance_scale,
            'coupling_weight': coupling_weight,
            'alignment_weight': alignment_weight,
            'target_density': target_density,
            'batch_size': batch_size,
            'num_batches': num_batches,
            'timestamp': datetime.now().isoformat(),
            'error': None,
            **guidance_stats,
            **eval_results
        }
        
    except Exception as e:
        print(f"Error in trial {trial_id}: {e}")
        results = {
            'trial_id': trial_id,
            'guidance_scale': guidance_scale,
            'coupling_weight': coupling_weight,
            'alignment_weight': alignment_weight,
            'target_density': target_density,
            'batch_size': batch_size,
            'num_batches': num_batches,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Test physics-aware magnetic guidance")
    parser.add_argument("--results-dir", type=str, default="physics_aware_results",
                      help="Directory for results")
    parser.add_argument("--num-trials", type=int, default=1,
                      help="Number of trials to run")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Parameter grid
    param_grid = {
        'guidance_scale': [1.0, 2.0, 5.0],
        'coupling_weight': [0.5, 1.0, 2.0],
        'alignment_weight': [0.3, 0.5, 1.0],
        'target_density': [0.9]  # Fixed for initial tests
    }
    
    # Generate parameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), values))
        for values in itertools.product(*param_grid.values())
    ]
    
    # Results storage
    all_results = []
    
    # Run trials
    for trial_id in range(args.num_trials):
        # Randomly select parameters for this trial
        params = random.choice(param_combinations)
        
        results = run_trial(
            trial_id=trial_id,
            results_dir=results_dir,
            **params
        )
        
        all_results.append(results)
        
        # Save results after each trial
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_dir / "results.csv", index=False)
        
        # Print summary
        print(f"\nTrial {trial_id} completed:")
        if results['error'] is None:
            print(f"Average density: {results['avg_density']:.6f} μB/Å³")
            print(f"Maximum density: {results['max_density']:.6f} μB/Å³")
            print(f"Average coupling score: {results['avg_coupling_score']:.2f}")
            print(f"Structures above 0.1 target: {results['pct_above_0.1_density']*100:.1f}%")
            print(f"Structures above 0.5 target: {results['pct_above_0.5_density']*100:.1f}%")
        else:
            print(f"Error: {results['error']}")
    
    print(f"\nAll trials completed. Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 