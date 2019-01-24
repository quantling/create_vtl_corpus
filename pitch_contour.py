import re
import os
from multiprocessing import Pool

def fit_f0(pitch_contour_dir, text_grid_dir, targetoptimizer_path='./bin/targetoptimizer', *, n_jobs=1):
    """
    Fits pitch contours and creates vocaltractlab gestures for the f0.

    Uses the targetoptimizer program.

    """
    all_pitch_tiers = os.listdir(pitch_contour_dir)

    old_ges = [os.path.splitext(f)[0] for f in all_pitch_tiers if f.endswith('.ges')]
    pitch_tiers = sorted([f for f in all_pitch_tiers if  f.endswith('.PitchTier')])

    commands = []
    for pitch_tier in pitch_tiers:
        base_name = os.path.splitext(pitch_tier)[0]
        if base_name in old_ges:
            # already created in older run
            continue

        if not os.path.exists(f"{text_grid_dir}/{base_name}.TextGrid"):
            print(f"WARNING: Does not exist: {text_grid_dir}/{base_name}.TextGrid")
            continue

        if not os.path.exists(f"{pitch_contour_dir}/{base_name}.PitchTier"):
            print(f"WARNING: Does not exist: {pitch_contour_dir}/{base_name}.PitchTier")
            continue

        # --m-range 1 confines the slope to +-1 which gives better extrapolation properties (but worse fitting)
        commands.append(f"{targetoptimizer_path} {text_grid_dir}/{base_name}.TextGrid {pitch_contour_dir}/{base_name}.PitchTier -g --m-range 1")

    with Pool(n_jobs) as pool:
        pool.map(os.system, commands)



def extract_pitch_tier(wav_dir, pitch_contour_dir, praat_script_path, *,
        n_jobs=1):

    base_names = sorted([os.path.splitext(ff)[0] for ff in os.listdir(wav_dir) if ff.endswith('.wav')])

    commands = [f"praat --run {praat_script_path} {wav_dir}/{base_name}.wav {pitch_contour_dir}/{base_name}.PitchTier" for base_name in base_names]

    # generate PitchTier
    with Pool(n_jobs) as pool:
        pool.map(os.system, commands)



def fix_all_ges(ges_dir, f0_dir, fixed_dir):
    orig_ges = sorted([os.path.splitext(f)[0] for f in os.listdir(ges_dir) if f.endswith('.ges')])

    for ii, ges_name in enumerate(orig_ges):
        src_path = f"{ges_dir}/{ges_name}.ges"
        f0_path = f"{f0_dir}/{ges_name}.ges"
        target_path = f"{fixed_dir}/{ges_name}.ges"

        print(ii, " ", target_path)

        if not os.path.exists(f0_path):
            print(f"WARNING: Does not exist: {f0_path} SKIP!")
            continue

        fix_ges(src_path, f0_path, target_path)



def fix_ges(orig_ges, f0_ges, fixed_ges):
    with open(orig_ges, "rt") as src_file:
        with open(fixed_ges, "wt") as target_file:
            within_f0_block = False
            for ii, line in enumerate(src_file):
                if line == '  <gesture_sequence type="f0-gestures" unit="st">\n':
                    within_f0_block = True
                    with open(f0_ges, 'rt') as f0_file:
                        f0_lines = f0_file.readlines()
                    # remove first two and last two lines:
                    f0_lines = f0_lines[2:-2]
                    f0_lines = [line.replace('\t', '  ') for line in f0_lines]
                    # really BAD hack!:
                    f0_lines = [line.replace('duration_s="0.000000"', 'duration_s="0.010000"') for line in f0_lines]
                    # BAD hack end.
                    fadein_line = f0_lines[0]
                    fadein_line = re.sub('slope="[0-9\.\-]*"', 'slope="0.000000"', fadein_line)
                    fadein_line = re.sub('duration_s="[0-9\.\-]*"', 'duration_s="0.030000"', fadein_line)
                    f0_lines.insert(0, fadein_line)
                    f0_lines.insert(0, '  <gesture_sequence type="f0-gestures" unit="st">\n')
                    f0_lines.append('  </gesture_sequence>\n')
                    target_file.writelines(f0_lines)
                    continue
                if within_f0_block and line == '  </gesture_sequence>\n':
                    within_f0_block = False
                    continue
                if within_f0_block:
                    continue
                target_file.write(line)

