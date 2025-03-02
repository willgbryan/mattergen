# Magnetic Density Optimization Learnings

## Changelog

### 2024-02-24: Pure Moment Maximization Results
- **Changes**:
  - Removed element-specific guidance
  - Increased power scaling (x⁵) for moments
  - Strengthened threshold bonuses
  - Increased per-atom moment bonuses
  - Doubled alignment bonus
- **Latest Results**:
  - New best total moment: 13.29 μB
  - Best composition: Ce₂Mn₆Al₄
  - Consistent high moments (>10 μB) in multiple trials
- **Key Insights**:
  - Element-agnostic approach still finds optimal compositions
  - Strong alignment correlates with high moments
  - Sweet spot found for guidance parameters:
    - Guidance scale: ~10000.0
    - Moment weight: ~1000.0

### 2024-02-24: Aggressive Reward Scaling Results
- **Changes**:
  - Switched to cubic scaling (x³) for moments
  - Added threshold bonuses (2x at 10μB, 4x at 20μB, 1.5x for high per-atom)
  - Increased guidance scales (2000-10000) and weights (500-2000)
- **Results**:
  - New best total moment: 11.47 μB (67% improvement)
  - Best per-atom moment: 1.15 μB/atom
  - First structure exceeding 10 μB threshold (25% rate at best config)
- **Best Configuration**:
  - Guidance scale: 10000.0
  - Moment weight: 1000.0
  - Average moment: 3.01 μB
- **Key Insights**:
  - Higher guidance scales (10000.0) significantly more effective
  - Sweet spot for moment weight around 1000.0
  - Cubic scaling with thresholds successfully driving toward higher moments
  - Model maintains reasonable volumes (160-323 Å³) even with aggressive guidance

### 2024-02-24: Atom-Only Guidance Results
- **Results**: Significant improvement in magnetic moments
  - Highest total moment: 6.85 μB
  - Best per-atom moment: 2.28 μB/atom
  - Multiple structures > 4 μB total moment
- **Key Findings**:
  - Higher moment weights (500.0) consistently produced better results
  - Guidance scale impact varies, with both 1000.0 and 2000.0 achieving high moments
  - Perfect alignment (1.0) maintained across all structures
  - Reasonable volume ranges (98-330 Å³) without explicit constraints
- **Best Configurations**:
  - Trial 3: 6.86 μB (guidance_scale=1000.0, moment_weight=100.0)
  - Trial 8: 6.85 μB (guidance_scale=2000.0, moment_weight=500.0)
  - Trial 2: 4.15 μB (guidance_scale=500.0, moment_weight=500.0)
- **Insights**:
  - Atom-only guidance appears more effective than multi-component approach
  - Model maintains reasonable structures without explicit volume control
  - Higher moment weights consistently lead to better performance

### 2024-02-24: Component-Specific Guidance Approach
- **Major Change**: Switched to atom-only guidance, removing position and cell rewards
- **Motivation**: Discovered MatterGen's separate guidance mechanisms for atoms, positions, and cell
- **Changes**:
  - Simplified reward to focus only on atomic_numbers rewards
  - Removed volume constraints and alignment bonuses
  - Adjusted guidance scales down (500-2000) for atom-specific control
  - Widened moment weight range (100-500) to explore atom selection impact
- **Status**: Pending experimental validation
- **Previous Best**: 0.205 μB/Å³ (from initial attempts)

### 2024-02-23: Multi-Component Guidance Attempt
- **Approach**: Simultaneous guidance of all structural components
- **Implementation**:
  - Complex reward with density, moment, and volume terms
  - Progressive penalties and bonuses
  - Per-atom volume constraints (8-20 Å³/atom)
- **Results**: 
  - Max density: 0.141 μB/Å³
  - More consistent but lower peak performance
  - Better average performance than initial attempts

### 2024-02-22: Initial Implementation
- **Approach**: Direct density optimization
- **Implementation**:
  - Complex multi-term reward structure
  - Focus on magnetic moment optimization
  - Heavy density penalties
- **Results**:
  - Best achieved: 0.205 μB/Å³
  - High variance in outcomes
  - Inconsistent performance

### 2024-02-28: Record-Breaking Moment Results
- **New Records**:
  - All-time best total moment: 27.95 μB (Gd₄S₅Cl₃)
  - Multiple structures exceeding 20 μB
  - Strong results from rare earth + transition metal combinations
- **Notable Structures**:
  - Gd₄S₅Cl₃: 27.95 μB (best overall)
  - Fe₆O₁₀F₂: 25.99 μB
  - Gd₂MnIrO₆: 19.06 μB
  - Pr₂Fe₈B₂: 17.15 μB
  - Eu₂Ni₆B₄: 13.57 μB
- **Key Insights**:
  - Rare earth elements (Gd, Pr, Eu) consistently produce high moments
  - Iron-rich compounds show excellent performance
  - Optimal compositions tend to have 8-12 atoms
  - Perfect alignment maintained across all structures
  - Higher guidance scales (15000-20000) producing better results
  - Moment weights of 1500-2500 most effective

### 2024-03-01: New Record - 42 μB Total Moment
- **New Records**:
  - All-time best total moment: 42.08 μB (Ba₁Gd₆Te₁₃)
  - Second best: 41.00 μB (Eu₆Hg₂P₈)
  - Multiple structures exceeding 25 μB
  - Strong performance from Gd and Eu compounds
- **Notable Structures**:
  - Ba₁Gd₆Te₁₃: 42.08 μB (perfect alignment)
  - Eu₆Hg₂P₈: 41.00 μB
  - Fe₆O₃F₉: 26.63 μB
  - Eu₄Ga₂Ge₄: 27.33 μB
  - Gd₄N₁: 28.10 μB (highest per-atom at 5.62 μB/atom)
- **Key Insights**:
  - Rare earth elements (Gd, Eu) continue to dominate high-moment structures
  - Perfect alignment (1.0) maintained across all trials
  - Higher guidance scales (25000-30000) producing stronger results
  - Moment weights of 2500-3500 most effective
  - Compact structures (8-10 atoms) achieving highest moments

## Target and Current State
- **Target**: 1.5 μB/Å³
- **Best Total Moment**: 42.08 μB (new record)
- **Best Per-Atom Moment**: 5.62 μB/atom (in Gd₄N₁)
- **Previous Best**: 27.95 μB
- **Recent Achievements**: 
  - Multiple structures exceeding 40 μB threshold
  - 50% improvement in total moment (27.95 → 42.08 μB)
  - Consistent high moments across different compositions
  - Perfect magnetic alignment maintained
  - Compact, high-moment structures emerging

## Key Insights

### 1. Reward Function Evolution
1. **Initial Attempts**: Complex multi-term rewards
   - Best achieved: 0.205 μB/Å³
   - High variance in outcomes
2. **Early Refinement**: Heavy density penalties
   - Max density ~0.098 μB/Å³
   - Too constrained
3. **Multi-Component**: Simultaneous guidance
   - Max density ~0.141 μB/Å³
   - More consistent but lower peak
4. **Atom-Only**: Focused guidance
   - First 10+ μB structures
   - Better exploration of composition space
5. **Pure Moment**: Element-agnostic approach
   - First 20+ μB structures
   - Record of 27.95 μB
   - Naturally finds optimal compositions

### 2. Parameter Sensitivity
- **Guidance Scale**:
  - Sweet spot around 15000-20000
  - Higher values more effective than previous ranges
  - Consistent performance above 12000
- **Moment Weight**:
  - Optimal range 1500-2500
  - Strong correlation with performance
  - Higher weights enable more aggressive exploration

### 3. Structural Patterns
- **Composition Balance**:
  - Rare earth elements (Gd, Pr, Eu) as moment anchors
  - Transition metals (Fe, Mn, Ni) for additional moments
  - Support elements (S, O, B) for stability
- **Size Effects**:
  - Optimal structures: 8-12 atoms
  - Volumes: 200-300 Å³
  - Perfect alignment maintained

## Key Challenges

### 1. Scale of the Problem
- Target density requires either:
  - Very high magnetic moments in normal volumes
  - Example: For 100 Å³ structure with 4 atoms
    - Need total moment = 150 μB (1.5 μB/Å³ × 100 Å³)
    - Per-atom moment ≈ 37.5 μB/atom
  - Or moderate moments in very compact volumes
    - Trade-off between stability and density

### 2. Reward Function Evolution

#### Initial Attempts
- Best achieved: 0.205 μB/Å³
- Complex reward structure with multiple terms
- Focus on direct magnetic moment optimization
- Challenge: Inconsistent results, high variance

#### Early Refinement
- Complex multi-term reward with exponential penalties
- Heavy penalties for low densities
- Strict volume control
- Results: Max density ~0.098 μB/Å³
- Challenge: Too punitive, preventing exploration

#### Latest Approach
- Simplified reward structure
- Progressive penalties and bonuses
- Per-atom volume constraints
- Results: Max density ~0.141 μB/Å³
- More consistent but still below best achieved
- Better average performance but lower peak

### 3. Component-Specific Guidance
- Key insight: MatterGen has separate guidance for:
  1. Atomic numbers (atom type selection)
  2. Positions (atomic coordinates)
  3. Cell parameters (lattice)
- Previous approaches tried to guide all components simultaneously
- New hypothesis: Focus solely on atom selection
  - Let base model handle positions and cell naturally
  - Reduce competing objectives
  - Clearer guidance signal for atom selection
  - May allow better exploration of magnetic configurations

### 4. Parameter Sensitivity

#### Guidance Scale Evolution
- Initial range: [50.0, 100.0, 200.0]
- Increased to: [100.0, 200.0, 500.0]
- Further increased to: [500.0, 1000.0, 2000.0]
- Latest adjustment: Back to [500.0, 1000.0, 2000.0] for atom-only guidance
- Observation: Need to recalibrate scales for component-specific guidance

#### Weight Balancing
- Previous best configuration:
  - Guidance scale: 200.0
  - Density weight: 10.0
  - Moment weight: 20.0
  - Volume weight: 0.2
- New simplified approach:
  - Only moment weight: [100.0, 200.0, 500.0]
  - Wider range to explore atom selection impact
  - No volume or density weights

## Key Insights

### 1. Physical Constraints
- Trade-off between:
  - Magnetic moment maximization
  - Volume minimization
  - Structure stability
- Model maintains reasonable structures even with very strong guidance
- Higher moments achievable than initially thought

### 2. Reward Design Principles
1. **Clarity**: Simple, direct rewards work better than complex ones
2. **Component Specificity**: Target specific aspects (atoms/positions/cell) rather than everything
3. **Physical Relevance**: Rewards should reflect physical relationships
4. **Aggressive Scaling**: Cubic scaling with thresholds more effective than quadratic
5. **Strong Guidance**: Much higher guidance scales (10000.0) effective when focused

### 3. Volume Control
- Previous: Explicit volume constraints
- Current: No volume constraints
- Finding: Model naturally maintains reasonable volumes (160-323 Å³)
- Validates hypothesis about base model's structural understanding

### 4. Moment Guidance
- Cubic scaling provides stronger emphasis on high moments
- Threshold bonuses create clear targets for optimization
- Sweet spot found for guidance parameters:
  - Guidance scale: ~10000.0
  - Moment weight: ~1000.0
- Perfect alignment maintained across all structures

### 5. Fundamental Control Question
- Previous concern: Too focused on density
- New approach: Verify basic control over magnetic properties
- Testing hypothesis that simpler, focused guidance might be more effective
- Separating concerns: atom selection vs. structural optimization

## Future Directions

### 1. Component-Specific Exploration
- Study effectiveness of atom-only guidance
- Consider staged approach:
  1. Optimize atom selection first
  2. Add position guidance if needed
  3. Consider cell guidance last
- Analyze which components most influence magnetic properties

### 2. Physical Insights
- Study high-moment atoms and their typical oxidation states
- Identify promising element combinations
- Consider chemical composition effects

### 3. Technical Improvements
- Calibrate guidance scales for atom-only control
- Study impact of reward normalization
- Consider alternative reward formulations for atom selection

### 4. Alternative Approaches
- Start from known magnetic elements
- Focus on specific chemical systems
- Consider element-specific rewards

## Open Questions

1. Will atom-only guidance produce better results than full guidance?
2. What is the right balance of guidance scale for atom selection?
3. Should we consider element-specific rewards?
4. How much do positions and cell parameters matter for magnetic properties?
5. Are we still hitting fundamental limits of the model's capabilities?
6. Could staged optimization (atoms → positions → cell) be more effective?

## Next Steps

1. **Parameter Refinement**:
   - Even finer grid around sweet spot
   - Explore batch size impact
   - Test longer generation runs

2. **Reward Engineering**:
   - Test higher power scaling (x⁶)
   - Adjust threshold levels
   - Fine-tune bonus multipliers

3. **Analysis**:
   - Study moment distribution patterns
   - Analyze volume-moment relationships
   - Track composition evolution

## Implementation Plan

### Reward Function
- Fifth power scaling base
- Progressive thresholds:
  - 2x at 5 μB
  - 3x at 10 μB
  - 4x at 13 μB
  - 5x at 15 μB
- Per-atom bonuses:
  - 2x above 1 μB/atom
  - 3x above 2 μB/atom
  - 4x above 3 μB/atom
- Double bonus for alignment > 0.8

### Parameter Grid
- Guidance scale: [9000, 10000, 11000]
- Moment weight: [900, 1000, 1100]
- Batch size: 12
- Batches per trial: 2

### Success Metrics
- Total magnetic moment
- Per-atom magnetic moments
- Alignment scores
- Structure stability
- Composition diversity

### Expected Outcomes
- More structures > 13 μB
- Higher average moments
- Better consistency
- New composition patterns

# Learnings from MatterGen Development

## Physics-Aware Magnetic Guidance

### Understanding the Core Diffusion Process
The diffusion process in MatterGen uses a predictor-corrector approach where:
1. A predictor estimates the next denoised state
2. A corrector refines this estimate (optional)
3. This process repeats through decreasing timesteps

### Key Insight: Direct Physical Guidance
Rather than using simple reward functions that only look at final properties, we can guide the diffusion process more effectively by:
1. Computing physics-based gradients at each denoising step
2. Directly influencing atomic positions and cell parameters
3. Using local magnetic interactions to guide structure formation

### Implementation Details
The enhanced magnetic guidance:
1. **Position Gradients**:
   - Evaluates magnetic coupling between atoms
   - Uses a Morse-like potential centered at optimal coupling distances
   - Weights gradients by magnetic moment magnitudes

2. **Cell Gradients**:
   - Guides cell parameters toward optimal density
   - Preserves crystal symmetry while scaling
   - Balances volume changes with magnetic coupling

3. **Score Modification**:
   - Directly modifies the score function at each step
   - Combines base diffusion scores with physical gradients
   - Allows exploration while maintaining magnetic constraints

### Benefits Over Previous Approach
1. More physically meaningful guidance
2. Better exploration of magnetic coupling mechanisms
3. Avoids bias toward specific structural types (e.g., rare earth mechanisms)
4. Smoother optimization landscape for the diffusion process

### Future Directions
1. Incorporate additional physical interactions
2. Add temperature-dependent magnetic coupling
3. Consider electronic structure effects
4. Explore adaptive guidance scaling based on timestep 