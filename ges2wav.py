import ctypes
import sys
import shutil
import os

VTL = ctypes.cdll.LoadLibrary(os.path.dirname(__file__) + '/bin/VocalTractLabApi.so')
speaker_file_name = ctypes.c_char_p((os.path.dirname(__file__) + '/JD2.speaker').encode())

name = sys.argv[1]
ges_dir = sys.argv[2]
wav_dir = sys.argv[3]
cp_dir = sys.argv[4]

gesture_file_name = ctypes.c_char_p(f'{ges_dir}/{name}.ges'.encode())
wav_file_name = ctypes.c_char_p(f'{wav_dir}/{name}.wav'.encode())
feedback_file_name = ctypes.c_char_p(f'{cp_dir}/{name}.txt'.encode())

failure = VTL.vtlGesToWav(speaker_file_name,  # input
                          gesture_file_name,  # input
                          wav_file_name,  # output
                          feedback_file_name)  # output

if failure != 0:
    #raise ValueError('Error in vtlGesToWav! Errorcode: %i' % failure)
    print('Error in vtlGesToWav! Errorcode: %i' % failure)
    print(f'move "{name}" to ./BAD/ folder')
    shutil.copy(f'{ges_dir}/{name}.ges', f'./BAD/{name}.ges')

