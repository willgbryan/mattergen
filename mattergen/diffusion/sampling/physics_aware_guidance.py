"""
Physics-aware guidance for magnetic property optimization in MatterGen.

This module provides enhanced diffusion guidance that incorporates physical principles
of magnetic interactions directly into the score function at each denoising step.
"""

import torch
import numpy as np
from typing import Optional, Dict, List
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector, BatchTransform
from pymatgen.core import Structure


class PhysicsAwareMagneticGuidance(GuidedPredictorCorrector):
    """
    Enhanced sampler for magnetic property guidance that directly influences
    the diffusion process through physics-aware gradients.
    """

    def __init__(
        self,
        *,
        guidance_scale: float,
        target_density: float = 0.9,  # Target in μB/Å³
        coupling_weight: float = 1.0,
        alignment_weight: float = 0.5,
        chgnet_model=None,
        remove_conditioning_fn: BatchTransform,
        keep_conditioning_fn: Optional[BatchTransform] = None,
        **kwargs,
    ):
        """
        Args:
            guidance_scale: Controls strength of property guidance
            target_density: Target magnetic density in μB/Å³
            coupling_weight: Weight for magnetic coupling term
            alignment_weight: Weight for moment alignment term
            chgnet_model: Pre-loaded CHGNet model
            remove_conditioning_fn: Function that removes conditioning
            keep_conditioning_fn: Function applied before evaluating conditional score
            **kwargs: Passed to parent class constructor
        """
        super().__init__(
            guidance_scale=guidance_scale,
            remove_conditioning_fn=remove_conditioning_fn,
            keep_conditioning_fn=keep_conditioning_fn,
            **kwargs
        )
        from chgnet.model.model import CHGNet
        self.chgnet_model = chgnet_model or CHGNet.load()
        self.target_density = target_density
        self.coupling_weight = coupling_weight
        self.alignment_weight = alignment_weight
        
        # Physics parameters
        self.optimal_coupling_distance = 2.5  # Å
        self.coupling_cutoff = 4.0  # Å
        self.coupling_width = 0.5  # Å (width of coupling potential)
        
        # Tracking for analysis
        self.structures_seen = 0
        self.best_moments: List[float] = []
        self.best_densities: List[float] = []

    def _compute_magnetic_coupling_gradient(self, distance: float) -> float:
        """
        Calculate gradient of magnetic coupling strength with respect to distance.
        Uses a Morse-like potential centered at optimal coupling distance.
        """
        dev = distance - self.optimal_coupling_distance
        coupling = -dev * np.exp(-dev**2 / (2 * self.coupling_width**2))
        return float(coupling)

    def _compute_position_gradients(self, structure: Structure, mag_moments: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradients for atomic positions to improve magnetic coupling.
        Returns gradients that guide atoms toward optimal magnetic coupling distances.
        """
        gradients = torch.zeros_like(torch.tensor(structure.cart_coords))
        magnetic_sites = [i for i, m in enumerate(mag_moments) if abs(m) > 0.1]
        
        for i in magnetic_sites:
            neighbors = structure.get_neighbors(structure[i], r=self.coupling_cutoff)
            for neighbor, dist in neighbors:
                j = neighbor.index
                if j in magnetic_sites:
                    # Direction vector from i to j
                    direction = (structure.cart_coords[j] - structure.cart_coords[i]) / dist
                    # Coupling gradient based on distance
                    coupling_grad = self._compute_magnetic_coupling_gradient(dist)
                    # Weight by magnetic moments
                    moment_factor = float(mag_moments[i] * mag_moments[j])
                    gradients[i] += torch.tensor(direction) * coupling_grad * moment_factor
        
        return gradients

    def _compute_cell_gradients(self, structure: Structure, mag_moments: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradients for cell parameters to optimize magnetic interactions.
        Returns gradients that guide cell parameters toward optimal density while preserving structure.
        """
        cell_gradients = torch.zeros((3, 3))
        magnetic_sites = [i for i, m in enumerate(mag_moments) if abs(m) > 0.1]
        
        if len(magnetic_sites) >= 2:
            # Get current cell parameters
            cell_matrix = torch.tensor(structure.lattice.matrix)
            
            # Calculate current magnetic density
            volume = structure.volume
            total_moment = sum(mag_moments)
            current_density = total_moment / volume
            
            # Guide cell parameters toward optimal density
            density_factor = self.target_density / (current_density + 1e-6)  # Avoid div by 0
            
            # Scale cell to approach target density while preserving angles
            scale_factor = torch.pow(density_factor, 1/3)  # Cubic root for 3D scaling
            target_cell = cell_matrix * scale_factor
            
            # Compute gradients toward target cell
            cell_gradients = (target_cell - cell_matrix) * 0.1  # Small step size
        
        return cell_gradients

    def _get_score(self, batch: dict, t: torch.Tensor) -> dict:
        """
        Override parent method to incorporate physics-aware magnetic guidance.
        
        Args:
            batch: Dictionary containing structure information
            t: Timesteps tensor
            
        Returns:
            Dictionary containing modified scores
        """
        # Get base scores from parent class
        base_scores = super()._get_score(batch, t)
        
        # Convert batch to structures
        structures = self._batch_to_structures(batch)
        position_gradients = []
        cell_gradients = []
        
        # Compute gradients for each structure
        for structure in structures:
            # Get magnetic predictions
            prediction = self.chgnet_model.predict_structure(structure)
            mag_moments = prediction['m']
            
            # Track statistics
            self.structures_seen += 1
            total_moment = float(sum(mag_moments))
            density = total_moment / structure.volume
            self.best_moments.append(total_moment)
            self.best_densities.append(density)
            
            # Compute gradients
            pos_grad = self._compute_position_gradients(structure, mag_moments)
            cell_grad = self._compute_cell_gradients(structure, mag_moments)
            
            position_gradients.append(pos_grad)
            cell_gradients.append(cell_grad)
        
        # Stack gradients
        position_gradients = torch.stack(position_gradients)
        cell_gradients = torch.stack(cell_gradients)
        
        # Scale gradients by guidance scale and weights
        if 'pos' in base_scores:
            base_scores['pos'] = base_scores['pos'] + self.guidance_scale * position_gradients
        if 'cell' in base_scores:
            base_scores['cell'] = base_scores['cell'] + self.guidance_scale * cell_gradients
        
        return base_scores

    def get_statistics(self) -> Dict[str, float]:
        """Return current optimization statistics."""
        if not self.best_moments:
            return {}
        
        return {
            'structures_seen': self.structures_seen,
            'best_moment': max(self.best_moments),
            'best_density': max(self.best_densities),
            'avg_moment': sum(self.best_moments) / len(self.best_moments),
            'avg_density': sum(self.best_densities) / len(self.best_densities)
        } 