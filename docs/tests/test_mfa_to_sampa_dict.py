import ctypes
import sys
import os
import contextlib
import time
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from create_vtl_corpus import create_corpus
from paule import util
import soundfile as sf
import shutil


vtl_worker = create_corpus.CreateCorpus("", language="de")


def test_mfa_to_sampa():
    VTL = create_corpus.util.VTL
    mfa_to_sampa_dict = vtl_worker.mfa_to_sampa_dict

    phones = mfa_to_sampa_dict.values()
    phone_durations = [0.1 for _ in phones]

    # write seg file
    rows = []
    for i, phone in enumerate(phones):
        row = "name = %s; duration_s = %f;" % (phone, phone_durations[i])
        rows.append(row)
    text = "\n".join(rows)
    path = os.path.join("tests", "clips")
    if not os.path.exists(path):
        os.mkdir(path=path)
    # delete this later
    if not os.path.exists(os.path.join(path, "temp_output")):
        os.mkdir(path=os.path.join(path, "temp_output"))
    seg_file_name = str(os.path.join(path, f"temp_output/all_phones.seg"))
    with open(seg_file_name, "w") as text_file:
        text_file.write(text)

    # get tract files and gesture score
    seg_file_name = ctypes.c_char_p(seg_file_name.encode())

    ges_file_name = str(os.path.join(path, f"temp_output/all_phones.ges"))
    ges_file_name = ctypes.c_char_p(ges_file_name.encode())

    VTL.vtlSegmentSequenceToGesturalScore(seg_file_name, ges_file_name)


#    tract_file_name = str(os.path.join(path, f"temp_output/target_audio_word_{word_index}.txt"))
#    c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())
#
#    util.VTL.vtlGesturalScoreToTractSequence(
#                                ges_file_name, c_tract_file_name
#                                                )
#    cps = util.read_cp(tract_file_name)
#
#    cp_norm = util.normalize_cp(cps)
#    cp_norms.append(cp_norm)


def test_each_phone():
    with open("tests/no_phoneme.pkl", "rb") as handle:
        no_phoneme = pickle.load(
            handle
        )  ## no phoneme is just standard noise, its generated by a durattion of 2 seconds
    with open("tests/no_phoneme_sr.pkl", "rb") as handle:
        no_phoneme_sr = pickle.load(handle)
    VTL = create_corpus.util.VTL
    mfa_to_sampa_dict = vtl_worker.mfa_to_sampa_dict

    phones = mfa_to_sampa_dict.values()
    duration = 2

    path = os.path.join("tests", "clips")
    if not os.path.exists(path):
        os.mkdir(path=path)
    # delete this later
    if not os.path.exists(os.path.join(path, "temp_output", "phones")):
        os.mkdir(path=os.path.join(path, "temp_output", "phones"))
    else:
        shutil.rmtree(os.path.join(path, "temp_output", "phones"))
        os.mkdir(path=os.path.join(path, "temp_output", "phones"))
    sf.write(
        f"tests/clips/temp_output/phones/no_phoneme.wav", no_phoneme, no_phoneme_sr
    )

    for i, phone in enumerate(phones):
        row = "name = %s; duration_s = %f;" % (phone, duration)

        text = row

        seg_file_name = str(os.path.join(path, f"temp_output/phones/{phone}.seg"))
        with open(seg_file_name, "w") as text_file:
            text_file.write(text)

        # get tract files and gesture score
        seg_file_name = ctypes.c_char_p(seg_file_name.encode())

        ges_file_name = str(os.path.join(path, f"temp_output/phones/{phone}.ges"))
        ges_file_name = ctypes.c_char_p(ges_file_name.encode())

        VTL.vtlSegmentSequenceToGesturalScore(seg_file_name, ges_file_name)

        tract_file_name = str(
            os.path.join(path, f"temp_output/phones/target_audio_word_{phone}.txt")
        )
        c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())

        util.VTL.vtlGesturalScoreToTractSequence(ges_file_name, c_tract_file_name)
        cps = util.read_cp(tract_file_name)
        wav_syn, wav_syn_sr = util.speak(cps)
        os.remove(tract_file_name)
        os.remove(ges_file_name.value.decode())
        os.remove(seg_file_name.value.decode())
        sf.write(f"tests/clips/temp_output/phones/{phone}.wav", wav_syn, wav_syn_sr)
        assert len(wav_syn) > 0
        if len(wav_syn) == len(no_phoneme):
            assert not (
                np.array_equal(wav_syn, no_phoneme)
            ), f"Phone: {phone} appears to be  no phoneme just standard noise"
        assert wav_syn is not None, f"Phone: {phone} is None"
