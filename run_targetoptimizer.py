import os

def fit_f0(pitch_contour_dir, text_grid_dir, targetoptimizer_path='targetoptimizer'):
    """
    Fits pitch contours and creates vocaltractlab gestures for the f0.

    Uses the targetoptimizer program.

    """
    all_pitch_tiers = os.listdir(pitch_contour_dir)

    old_ges = [os.path.splitext(f)[0] for f in all_pitch_tiers if f.endswith('.ges')]
    pitch_tiers = sorted([f for f in all_pitch_tiers if  f.endswith('.PitchTier')])

    for pitch_tier in pitch_tiers:
        base_name = os.path.splitext(pitch_tier)[0]
        if base_name in old_ges:
            # already created in older run
            continue

        if not os.path.exists(f"{text_grid_dir}/{base_name}.TextGrid"):
            print(f"WARNING: Does not exist: {text_grid_dir}/{base_name}.TextGrid")
            continue

        if not os.path.exists(f"{pitch_contour_dir}/{base_name}.PitchTier"):
            print(f"WARNING: Does not exist: {pitch_tier_dir}/{base_name}.PitchTier")
            continue

        # --m-range 1 confines the slope to +-1 which gives better extrapolation properties (but worse fitting)
        os.system(f"./targetoptimizer {text_grid_dir}/{base_name}.TextGrid {pitch_contour_dir}/{base_name}.PitchTier -g --m-range 1")

