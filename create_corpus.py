
import os
import shutil
from multiprocessing import Pool

from . import tgwav2utt, pitch_contour, sampa2ges

def create_corpus(version, *, geco_path=None):
    """
    creates the vtl corpus.

    """
    if not geco_path:
        raise ValueError("You need to specify the path where all GECO corpus files live that you have downloaded seperately.")
    VERSION = version
    GECO_PATH = geco_path
    WAV_ORIG_DIR = f"vtl_corpus{VERSION}/wav_original"
    WAV_SYNTH_DIR = f"vtl_corpus{VERSION}/wav_synthesized"
    PLAIN_GES_DIR = f"vtl_corpus{VERSION}/plain_gestures"
    FIXED_GES_DIR = f"vtl_corpus{VERSION}/fixed_gestures"
    CP_DIR = f"vtl_corpus{VERSION}/control_parameters"
    PITCH_DIR = f"vtl_corpus{VERSION}/pitch_contours"
    TG_DIR = f"vtl_corpus{VERSION}/text_grids"
    UTT_NAME = f"vtl_corpus{VERSION}/corpus_sampa{VERSION}.utt"


    ## create folders
    #os.makedirs(WAV_ORIG_DIR)
    #os.makedirs(WAV_SYNTH_DIR)
    #os.makedirs(PLAIN_GES_DIR)
    #os.makedirs(FIXED_GES_DIR)
    #os.makedirs(CP_DIR)
    #os.makedirs(PITCH_DIR)
    #os.makedirs(TG_DIR)
    #
    #
    ## text grid + wave -> utterance, wav, text grids
    #base_names = [os.path.splitext(f)[0] for f in os.listdir(f"{GECO_PATH}/textgrids") if f.endswith('.textGrid')]
    #ii = 0
    #for base_name in base_names:
    #    tg_name = f"{GECO_PATH}/textgrids/{base_name}.textGrid"
    #    wav_name = f"{GECO_PATH}/wav/{base_name}.wav"
    #    ii = tgwav2utt.create_utterance_split_wave(tg_name, wav_name, UTT_NAME, WAV_ORIG_DIR, TG_DIR, ii=ii)
    #
    #
    ## extract pitch contours with praat
    #pitch_contour.extract_pitch_tier(os.path.abspath(WAV_ORIG_DIR), os.path.abspath(PITCH_DIR), praat_script_path="./extractpitch.praat",  n_jobs=8)
    #
    #pitch_contour.fit_f0(os.path.abspath(PITCH_DIR), os.path.abspath(TG_DIR), "./bin/targetoptimizer", n_jobs=8)
    #
    #
    ## heuristacally create ges files
    #sampa2ges.sampa_to_ges(UTT_NAME, PLAIN_GES_DIR, phone_attributes='./phone_attributes.txt')
    #
    #
    ## insert f0 fit into ges files
    #pitch_contour.fix_all_ges(PLAIN_GES_DIR, PITCH_DIR, FIXED_GES_DIR)



    # synthesize wav
    ges_files = [os.path.splitext(f)[0] for f in os.listdir(FIXED_GES_DIR) if f.endswith('.ges')]

    commands = []
    for ges_name in ges_files:
        commands.append(f'python {os.path.dirname(__file__)}/ges2wav.py '
                        f'"{ges_name}" "{os.path.abspath(FIXED_GES_DIR)}" '
                        f'"{os.path.abspath(WAV_SYNTH_DIR)}"  '
                        f'"{os.path.abspath(CP_DIR)}"')

    with Pool(8) as pool:
        pool.map(os.system, commands)

