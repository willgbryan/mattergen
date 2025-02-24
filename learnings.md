# Magnetic Density Optimization Learnings

## Changelog

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

## Target and Current State
- **Target**: 1.5 μB/Å³
- **Best Achieved Overall**: 0.205 μB/Å³ (~13.7% of target)
- **Recent Performance**: 0.140783 μB/Å³ (~9.4% of target)
- **Average Performance**: 0.005722 ± 0.010544 μB/Å³

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
- Need to balance these competing factors

### 2. Reward Design Principles
1. **Clarity**: Simple, direct rewards work better than complex ones
2. **Component Specificity**: Target specific aspects (atoms/positions/cell) rather than everything
3. **Physical Relevance**: Rewards should reflect physical relationships
4. **Focus**: Single clear objective may be better than multiple competing ones

### 3. Volume Control
- Previous: Explicit volume constraints
- New approach: Let model's natural preferences guide volume
- Hypothesis: Base model's understanding of reasonable structures may be sufficient

### 4. Moment Guidance
- Focus solely on total moment through atom selection
- Quadratic reward to emphasize higher moments
- Normalized by number of atoms for fair comparison
- Removed alignment bonuses to simplify guidance

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

1. **Immediate Priority**: Test atom-only guidance approach
   - Run new tuning script with simplified reward
   - Focus on atom selection impact
   - Compare results with previous approaches
   - Analyze which elements are being selected

2. Based on atom-only results:
   - If successful: Fine-tune guidance parameters
   - If unsuccessful: Consider adding position guidance
   - Analyze patterns in selected elements

3. Study successful structures:
   - Element combinations
   - Coordination environments
   - Volume distributions

4. Consider element-specific guidance:
   - Identify promising magnetic elements
   - Create targeted rewards for specific elements
   - Study chemical composition patterns

## Implementation Plan for Atom-Only Guidance

### Reward Function
- Simple quadratic reward for total moment
- Normalized by number of atoms
- No position or cell rewards
- Focus on element selection

### Parameter Ranges
- Guidance scale: [500.0, 1000.0, 2000.0]
- Moment weight: [100.0, 200.0, 500.0]
- No volume or density parameters

### Success Metrics
- Total magnetic moment
- Per-atom magnetic moments
- Element distribution analysis
- Structure stability (secondary metric)

### Expected Outcomes
- Better understanding of element selection impact
- Clearer picture of achievable magnetic moments
- Insight into whether structural guidance is necessary 