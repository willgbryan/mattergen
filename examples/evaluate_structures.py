"""
Script for evaluating previously generated structures from MatterGen.

This script:
1. Loads structures from a zip file or directory
2. Evaluates magnetic properties (moments and densities)
3. Evaluates stability (formation energies)
4. Prints detailed statistics
"""

import os
from pathlib import Path
import torch
import zipfile
import tempfile
import argparse
from chgnet.model.model import CHGNet
from pymatgen.core import Structure

def evaluate_structures(
    input_path: Path,
    is_zip: bool = True,
    chgnet_model = None
) -> dict:
    """
    Evaluate magnetic and stability properties of structures.
    
    Args:
        input_path: Path to zip file or directory containing CIF files
        is_zip: Whether the input_path points to a zip file
        chgnet_model: Pre-loaded CHGNet model (will load if None)
        
    Returns:
        Dictionary containing evaluation results and statistics
    """
    # Load CHGNet if not provided
    if chgnet_model is None:
        print("\nLoading CHGNet model...")
        chgnet_model = CHGNet.load()
    
    def process_cif_files(cif_dir: Path):
        cif_files = list(cif_dir.glob("*.cif"))
        print(f"\nFound {len(cif_files)} CIF files")
        
        if not cif_files:
            raise ValueError("No CIF files found!")
        
        structures = []
        structure_files = []  # Keep track of filenames
        for f in cif_files:
            try:
                structure = Structure.from_file(f)
                structures.append(structure)
                structure_files.append(f.name)
            except Exception as e:
                print(f"Error loading structure from {f}: {e}")
        
        print(f"Successfully loaded {len(structures)} structures")
        
        if not structures:
            raise ValueError("No valid structures were loaded!")
        
        return structures, structure_files
    
    # Handle zip file or directory input
    if is_zip:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\nExtracting CIF files from {input_path}")
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            structures, structure_files = process_cif_files(Path(temp_dir))
    else:
        structures, structure_files = process_cif_files(input_path)
    
    # Evaluate properties
    moments = []
    densities = []
    energies = []
    detailed_results = []
    
    print("\nEvaluating structures...")
    for i, (structure, filename) in enumerate(zip(structures, structure_files)):
        try:
            # Get the prediction
            prediction = chgnet_model.predict_structure(structure)
            
            # Get magnetic moments (stored under 'm' key)
            mag_moments = prediction['m']  # This is per-atom array
            mag_moment = float(sum(mag_moments))  # Total magnetic moment
            
            # Get energy and volume
            volume = structure.volume
            formation_energy = float(prediction['e']) / len(structure)
            mag_density = mag_moment / volume
            
            moments.append(mag_moment)
            densities.append(mag_density)
            energies.append(formation_energy)
            
            result = {
                'index': i,
                'filename': filename,
                'volume': volume,
                'magnetic_moment': mag_moment,
                'magnetic_density': mag_density,
                'formation_energy': formation_energy,
                'per_atom_moments': mag_moments.tolist(),
                'num_atoms': len(structure)
            }
            detailed_results.append(result)
            
            print(f"\nStructure {i} ({filename}) evaluation:")
            print(f"  Number of atoms: {len(structure)}")
            print(f"  Volume: {volume:.3f} Å³")
            print(f"  Total magnetic moment: {mag_moment:.3f} μB")
            print(f"  Magnetic density: {mag_density:.6f} μB/Å³")
            print(f"  Per-atom moments: {mag_moments.tolist()}")
            print(f"  Formation energy: {formation_energy:.3f} eV/atom")
            
        except Exception as e:
            print(f"Error evaluating structure {i} ({filename}):")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {str(e)}")
            if hasattr(e, '__traceback__'):
                import traceback
                print("  Traceback:")
                traceback.print_tb(e.__traceback__)
    
    if not moments or not energies:
        print("\nNo valid predictions were made. Debugging information:")
        print(f"Number of structures processed: {len(structures)}")
        print(f"Number of magnetic moments: {len(moments)}")
        print(f"Number of formation energies: {len(energies)}")
        raise ValueError("No valid property predictions!")
    
    # Convert to tensors for statistics
    moments = torch.tensor(moments)
    densities = torch.tensor(densities)
    energies = torch.tensor(energies)
    
    # Compute statistics
    stats = {
        'magnetic': {
            'moment': {
                'mean': moments.mean().item(),
                'std': moments.std().item(),
                'min': moments.min().item(),
                'max': moments.max().item()
            },
            'density': {
                'mean': densities.mean().item(),
                'std': densities.std().item(),
                'min': densities.min().item(),
                'max': densities.max().item()
            }
        },
        'stability': {
            'formation_energy': {
                'mean': energies.mean().item(),
                'std': energies.std().item(),
                'min': energies.min().item(),
                'max': energies.max().item()
            }
        }
    }
    
    return {
        'statistics': stats,
        'detailed_results': detailed_results
    }

def print_evaluation_results(results: dict):
    """Print formatted evaluation results."""
    stats = results['statistics']
    
    print("\nMagnetic Properties:")
    print(f"Average magnetic moment: {stats['magnetic']['moment']['mean']:.2f} μB")
    print(f"Std of magnetic moments: {stats['magnetic']['moment']['std']:.2f} μB")
    print(f"Min/Max magnetic moment: {stats['magnetic']['moment']['min']:.2f}/{stats['magnetic']['moment']['max']:.2f} μB")
    
    print(f"\nMagnetic Density Properties:")
    print(f"Average magnetic density: {stats['magnetic']['density']['mean']:.6f} μB/Å³")
    print(f"Std of magnetic densities: {stats['magnetic']['density']['std']:.6f} μB/Å³")
    print(f"Min/Max magnetic density: {stats['magnetic']['density']['min']:.6f}/{stats['magnetic']['density']['max']:.6f} μB/Å³")
    
    print("\nStability Properties:")
    print(f"Average formation energy: {stats['stability']['formation_energy']['mean']:.3f} eV/atom")
    print(f"Std of formation energies: {stats['stability']['formation_energy']['std']:.3f} eV/atom")
    print(f"Min/Max formation energy: {stats['stability']['formation_energy']['min']:.3f}/{stats['stability']['formation_energy']['max']:.3f} eV/atom")

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated structures from MatterGen")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to zip file or directory containing CIF files"
    )
    parser.add_argument(
        "--directory",
        action="store_true",
        help="Treat input_path as directory instead of zip file"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return
    
    try:
        results = evaluate_structures(
            input_path=input_path,
            is_zip=not args.directory
        )
        print_evaluation_results(results)
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main() 