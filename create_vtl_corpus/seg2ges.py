import ctypes
import os

VTL = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/vocaltractlab_api/VocalTractLabApi.so')
#speaker_file_name = ctypes.c_char_p((os.path.dirname(__file__) + '/vocaltractlab_api/CK_female.speaker').encode())
speaker_file_name = ctypes.c_char_p((os.path.dirname(__file__) + '/vocaltractlab_api/JD2.speaker').encode())


def seg_to_ges(seg_dir, ges_dir):

    failure = VTL.vtlInitialize(speaker_file_name)
    if failure != 0:
        raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)

    seg_files = sorted([os.path.splitext(f)[0] for f in os.listdir(seg_dir) if f.endswith('.seg')])

    for name in seg_files:
        segment_file_name = ctypes.c_char_p(f'{seg_dir}/{name}.seg'.encode())
        gesture_file_name = ctypes.c_char_p(f'{ges_dir}/{name}.ges'.encode())

        failure = VTL.vtlSegmentSequenceToGesturalScore(segment_file_name, gesture_file_name)
        if failure != 0:
            print(name)
            raise ValueError('Error in vtlSegmentSequenceToGesturalScore! Errorcode: %i' % failure)

    VTL.vtlClose()

