import os
import re
#import shutil

PITCH_TIER_DIR = 'pitch_tier'
GES_DIR = 'ges'
TARGET_DIR = 'ges_f0optim'

all_pitch_tiers = os.listdir(PITCH_TIER_DIR)

f0_ges = [os.path.splitext(f)[0] for f in all_pitch_tiers if f.endswith('.ges')]
orig_ges = [os.path.splitext(f)[0] for f in os.listdir(GES_DIR) if f.endswith('.ges')]

for ges_name in orig_ges:
    src_path = f"{GES_DIR}/{ges_name}.ges"
    stripped_name = ges_name.strip(".")  # targetoptimizer strips "."
    f0_path = f"{PITCH_TIER_DIR}/{stripped_name}.ges"
    target_path = f"{TARGET_DIR}/{ges_name}.ges"
    if not os.path.exists(f0_path):
        #shutil.copy(src_path, target_path)
        print(f"WARNING: Does not exist: {f0_path} SKIP!")
        continue
    with open(src_path, "rt") as src_file:
        with open(target_path, "wt") as target_file:
            within_f0_block = False
            for ii, line in enumerate(src_file):
                if line == '  <gesture_sequence type="f0-gestures" unit="st">\n':
                    within_f0_block = True
                    with open(f0_path, 'rt') as f0_file:
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

