import os

PITCH_TIER_DIR = 'pitch_tier'
TEXT_GRID_DIR = 'textgrids'

all_pitch_tiers = os.listdir(PITCH_TIER_DIR)

old_ges = [os.path.splitext(f)[0] for f in all_pitch_tiers if f.endswith('.ges')]
pitch_tiers = sorted([f for f in all_pitch_tiers if  f.endswith('.PitchTier')])

for pitch_tier in pitch_tiers:
    base_name = os.path.splitext(pitch_tier)[0]
    if base_name in old_ges:
        # already created in older run
        continue

    if not os.path.exists(f"{TEXT_GRID_DIR}/{base_name}.TextGrid"):
        print(f"WARNING: Does not exist: {TEXT_GRID_DIR}/{base_name}.TextGrid")
        continue

    if not os.path.exists(f"{PITCH_TIER_DIR}/{base_name}.PitchTier"):
        print(f"WARNING: Does not exist: {PITCH_TIER_DIR}/{base_name}.PitchTier")
        continue

    os.system(f"./targetoptimizer {TEXT_GRID_DIR}/{base_name}.TextGrid {PITCH_TIER_DIR}/{base_name}.PitchTier -g --m-range 1")


