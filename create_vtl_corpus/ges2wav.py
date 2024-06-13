import ctypes
import sys
import shutil
import os

VTL = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/vocaltractlab_api/VocalTractLabApi.so')
#speaker_file_name = ctypes.c_char_p((os.path.dirname(__file__) + '/vocaltractlab_api/CK_female.speaker').encode())
speaker_file_name = ctypes.c_char_p((os.path.dirname(__file__) + '/vocaltractlab_api/JD2.speaker').encode())

name = sys.argv[1]
ges_dir = sys.argv[2]
wav_dir = sys.argv[3]
cp_dir = sys.argv[4]

gesture_file_name = ctypes.c_char_p(f'{ges_dir}/{name}.ges'.encode())
wav_file_name = ctypes.c_char_p(f'{wav_dir}/{name}.wav'.encode())
tract_sequence_file_name = ctypes.c_char_p(f'{cp_dir}/{name}.txt'.encode())

# API 2.2
#failure = VTL.vtlGesToWav(speaker_file_name,  # input
#                          gesture_file_name,  # input
#                          wav_file_name,  # output
#                          feedback_file_name)  # output

# API 2.3
failure = VTL.vtlInitialize(speaker_file_name)
if failure != 0:
    raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)

failure = VTL.vtlGesturalScoreToTractSequence(gesture_file_name, tract_sequence_file_name)
if failure != 0:
    raise ValueError('Error in vtlGesturalScoreToTractSequence! Errorcode: %i' % failure)

failure = VTL.vtlTractSequenceToAudio(tract_sequence_file_name, wav_file_name, None, None)
if failure != 0:
    #raise ValueError('Error in vtlGesToWav! Errorcode: %i' % failure)
    print('Error in vtlTractSequenceToAudio! Errorcode: %i' % failure)
    print(f'move "{name}" to ./BAD/ folder')
    shutil.copy(f'{ges_dir}/{name}.ges', f'./BAD/{name}.ges')

