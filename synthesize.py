import os

GES_DIR = 'ges_f0optim'
WAV_DIR = 'ges_f0optim'
FEEDBACK_DIR = 'ges_f0optim'

orig_ges = [os.path.splitext(f)[0] for f in os.listdir(GES_DIR) if f.endswith('.ges')]

for ges_name in orig_ges:
    os.system(f'python ges2wav.py {ges_name}')

