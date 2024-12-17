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
    """
    This test checks if the mfa_to_sampa_dict is working and produces a phoneme.
    It does not check if the phoneme is acutally accepted by VTL
    """
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
    path = os.path.join("docs", "tests", "clips")
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
    """
    This test checks if each phoneme in the mfa_to_sampa_dict is actually a phoneme
    and not just silent noise since VTL will not give an error if the phoneme is not a phoneme
    """

    def create_phoneme(phone, duration, phones_folder_path, VTL):
        "This function creates a phoneme and returns the wav file and the sample rate"
        row = "name = %s; duration_s = %f;" % (phone, duration)

        text = row

        seg_file_name = str(os.path.join(phones_folder_path, f"{phone}.seg"))
        with open(seg_file_name, "w") as text_file:
            text_file.write(text)

        # get tract files and gesture score
        seg_file_name = ctypes.c_char_p(seg_file_name.encode())

        ges_file_name = str(os.path.join(phones_folder_path, f"{phone}.ges"))
        ges_file_name = ctypes.c_char_p(ges_file_name.encode())

        VTL.vtlSegmentSequenceToGesturalScore(seg_file_name, ges_file_name)

        tract_file_name = str(
            os.path.join(phones_folder_path, f"target_audio_word_{phone}.txt")
        )
        c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())

        util.VTL.vtlGesturalScoreToTractSequence(ges_file_name, c_tract_file_name)
        cps = util.read_cp(tract_file_name)
        wav_syn, wav_syn_sr = util.speak(cps)
        os.remove(tract_file_name)
        os.remove(ges_file_name.value.decode())
        os.remove(seg_file_name.value.decode())
        # delete the non human readable files
        sf.write(os.path.join(phones_folder_path, f"{phone}.wav"), wav_syn, wav_syn_sr)
        return wav_syn, wav_syn_sr

    # Define all the paths here to avoid repeating the same code
    path_no_phoneme = os.path.join("docs", "tests", "no_phoneme.pkl")
    path_no_phoneme_sr = os.path.join("docs", "tests", "no_phoneme_sr.pkl")

    path = os.path.join("docs", "tests", "clips")
    if not os.path.exists(path):
        os.mkdir(path=path)
    # delete this later
    phones_folder_path = os.path.join(path, "temp_output", "phones")
    if not os.path.exists(phones_folder_path):
        os.mkdir(path=phones_folder_path)
    else:
        shutil.rmtree(phones_folder_path)
        os.mkdir(path=phones_folder_path)

    VTL = create_corpus.util.VTL
    mfa_to_sampa_dict = vtl_worker.mfa_to_sampa_dict

    phones = mfa_to_sampa_dict.values()
    duration = 2  # seconds , since this seems to be long enough to check if the phoneme is just silent, however this is not a guarantee and the test can take quite a while
    if not os.path.exists(path_no_phoneme):
        no_phoneme, no_phoneme_sr = create_phoneme(
            "bʲ", duration, phones_folder_path, VTL
        )
        with open(path_no_phoneme, "wb") as handle:
            pickle.dump(no_phoneme, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path_no_phoneme_sr, "wb") as handle:
            pickle.dump(no_phoneme_sr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        os.path.join(
            path_no_phoneme,
        ),
        "rb",
    ) as handle:
        no_phoneme = pickle.load(
            handle
        )  ## no phoneme is just standard noise, its generated by a durattion of 2 seconds

    with open(path_no_phoneme_sr, "rb") as handle:
        no_phoneme_sr = pickle.load(handle)

    sf_no_phoneme_path = os.path.join(phones_folder_path, "no_phoneme.wav")
    sf.write(
        sf_no_phoneme_path, no_phoneme, no_phoneme_sr
    )  # this is the silent noise to compare against

    for i, phone in enumerate(phones):  # now the actual test

        wav_syn, wav_syn_sr = create_phoneme(phone, duration, phones_folder_path, VTL)
        # ypou can listen to the phonemes to check if they are what you expect
        assert len(wav_syn) > 0
        if len(wav_syn) == len(no_phoneme):
            assert not (
                np.array_equal(wav_syn, no_phoneme)
            ), f"Phone: {phone} appears to be  no phoneme just standard noise"
        assert wav_syn is not None, f"Phone: {phone} is None"
    # we might think about deleting the files after the test
