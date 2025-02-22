from typing import Dict, Optional, Union
from mattergen.generator import CrystalGenerator
from mattergen.diffusion.sampling.magnetic_guidance import PropertyGuidedPredictorCorrector
from mattergen.diffusion.sampling.reward_functions import (
    BaseRewardFunction,
    MagneticRewardFunction,
    CompositeRewardFunction
)

def create_property_guided_generator(
    generator: CrystalGenerator,
    reward_function: Union[BaseRewardFunction, Dict[str, BaseRewardFunction]],
    guidance_scale: float = 1.0,
    reward_weights: Optional[Dict[str, float]] = None
) -> CrystalGenerator:
    """
    Create a CrystalGenerator that uses property guidance during generation.
    
    Args:
        generator: Base CrystalGenerator instance
        reward_function: Either a single BaseRewardFunction or a dictionary of
            named reward functions to combine
        guidance_scale: Controls strength of property guidance
        reward_weights: Optional weights for combining multiple reward functions.
            Only used if reward_function is a dictionary.
        
    Returns:
        Modified CrystalGenerator with property guidance
    """
    # Handle multiple reward functions
    if isinstance(reward_function, dict):
        reward_function = CompositeRewardFunction(
            reward_functions=reward_function,
            weights=reward_weights
        )
    
    # Get the original sampler config
    cfg = generator.cfg
    
    # Create new property guided sampler
    sampler = PropertyGuidedPredictorCorrector(
        guidance_scale=guidance_scale,
        reward_function=reward_function,
        remove_conditioning_fn=cfg.sampler.remove_conditioning_fn,
        keep_conditioning_fn=cfg.sampler.keep_conditioning_fn,
        diffusion_module=cfg.sampler.diffusion_module,
        predictor_partials=cfg.sampler.predictor_partials,
        corrector_partials=cfg.sampler.corrector_partials,
        device=cfg.sampler.device,
        n_steps_corrector=cfg.sampler.n_steps_corrector,
        N=cfg.sampler.N,
    )
    
    # Update the generator's sampler
    generator._model.sampler = sampler
    
    return generator


def create_magnetic_guided_generator(
    generator: CrystalGenerator,
    target_magnetic_moment: float,
    guidance_scale: float = 1.0,
    chgnet_model=None
) -> CrystalGenerator:
    """
    Convenience function to create a magnetically-guided generator.
    This is equivalent to using create_property_guided_generator with a MagneticRewardFunction.
    
    Args:
        generator: Base CrystalGenerator instance
        target_magnetic_moment: Target magnetic moment (in μB) to guide towards
        guidance_scale: Controls strength of magnetic property guidance
        chgnet_model: Pre-loaded CHGNet model, will load default if None
        
    Returns:
        Modified CrystalGenerator with magnetic guidance
    """
    reward_function = MagneticRewardFunction(
        target_magnetic_moment=target_magnetic_moment,
        chgnet_model=chgnet_model
    )
    
    return create_property_guided_generator(
        generator=generator,
        reward_function=reward_function,
        guidance_scale=guidance_scale
    )
