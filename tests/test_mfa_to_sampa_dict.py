import ctypes
import sys
import os
import contextlib

sys.path.insert(0, os.path.abspath("."))
from create_vtl_corpus import create_corpus


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
