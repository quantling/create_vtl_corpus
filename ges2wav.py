import ctypes
import sys
import shutil

VTL = ctypes.cdll.LoadLibrary('./VocalTractLabApi.so')
speaker_file_name = ctypes.c_char_p(b'JD2.speaker')

name = sys.argv[1]

gesture_file_name = ctypes.c_char_p(f'./ges_f0optim/{name}.ges'.encode())
wav_file_name = ctypes.c_char_p(f'./wav_synth/{name}.wav'.encode())
feedback_file_name = ctypes.c_char_p(f'./feedback/{name}.txt'.encode())

failure = VTL.vtlGesToWav(speaker_file_name,  # input
                          gesture_file_name,  # input
                          wav_file_name,  # output
                          feedback_file_name)  # output

if failure != 0:
    #raise ValueError('Error in vtlGesToWav! Errorcode: %i' % failure)
    print('Error in vtlGesToWav! Errorcode: %i' % failure)
    print(f'move "{name}" to ./BAD/ folder')
    shutil.copy(f'./ges_f0optim/{name}.ges', f'./BAD/{name}.ges')

