import torch
from typing import Optional
from mattergen.diffusion.sampling.classifier_free_guidance import GuidedPredictorCorrector
from mattergen.diffusion.sampling.classifier_free_guidance import BatchTransform
from mattergen.diffusion.sampling.reward_functions import BaseRewardFunction

class PropertyGuidedPredictorCorrector(GuidedPredictorCorrector):
    """
    Generic sampler for property-guided generation using custom reward functions.
    """

    def __init__(
        self,
        *,
        guidance_scale: float,
        reward_function: BaseRewardFunction,
        remove_conditioning_fn: BatchTransform,
        keep_conditioning_fn: Optional[BatchTransform] = None,
        **kwargs,
    ):
        """
        Args:
            guidance_scale: Controls strength of property guidance
            reward_function: Instance of BaseRewardFunction to compute rewards
            remove_conditioning_fn: Function that removes conditioning from the data
            keep_conditioning_fn: Function applied before evaluating conditional score
            **kwargs: Passed to parent class constructor
        """
        super().__init__(
            guidance_scale=guidance_scale,
            remove_conditioning_fn=remove_conditioning_fn,
            keep_conditioning_fn=keep_conditioning_fn,
            **kwargs
        )
        self.reward_function = reward_function

    def _get_score(self, batch: dict, t: torch.Tensor) -> dict:
        """
        Override parent method to incorporate property guidance via reward function.
        
        Args:
            batch: Dictionary containing structure information
            t: Timesteps tensor
            
        Returns:
            Dictionary containing scores
        """
        # Get base scores from parent class
        base_scores = super()._get_score(batch, t)
        
        # Compute reward using provided reward function
        reward = self.reward_function.compute_reward(batch)
        
        # Scale and reshape reward to match score dimensions
        for key in base_scores:
            if key in ['pos', 'cell']:  # Apply to continuous variables
                scaled_reward = (
                    reward.view(-1, 1, 1) *  # For pos
                    torch.ones_like(base_scores[key])
                )
                base_scores[key] = base_scores[key] + scaled_reward
                
        return base_scores
