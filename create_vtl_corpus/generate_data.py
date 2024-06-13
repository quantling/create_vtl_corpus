

import ctypes

import contextlib

import os
import pickle
import subprocess

from praatio import textgrid
import numpy as np
import pandas as pd
import soundfile as sf
from paule import util
import fasttext.util
import shutil
from tqdm import tqdm
import logging

logging.disable()

FASTTEXT_EMBEDDINGS = fasttext.load_model('cc.de.300.bin')

VTL_NEUTRAL_TRACT = np.array([1.0, -4.75, 0.0, -2.0, -0.07, 0.95, 0.0, -0.1, -0.4, -1.46, 3.5, -1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
VTL_NEUTRAL_TRACT.shape = (1, 19)

VTL_NEUTRAL_GLOTTIS = np.array([120.0, 8000.0, 0.01, 0.02, 0.05, 1.22, 1.0, 0.05, 0.0, 25.0, -10.0])
VTL_NEUTRAL_GLOTTIS.shape = (1, 11)


"""sampa_convert_dict = {
    'etu':'@',
    'atu':'6',
    'al':'a:',
    'el':'e:',
    'il':'i:',
    'ol':'o:',
    'ul':'u:',
    'oel':'2:',
    'uel':'y:',
    'ae':'E:',
    'oe':'9',
    'ue':'Y',
    'ng':'N',
    'eU':'OY'
}"""
with open('resources/sampa_ipa_dict.pkl', 'rb') as handle:
    sampa_convert_dict = pickle.load(handle)
sampa_convert_dict["c"] = "k"  # TODO bad quick fix

#path is the path where we have the mozilla common voice data
PATH = 'new_data'

# store inputs, validatet.tsv is from mozilla common voice 
TEXT = pd.read_csv(os.path.join(PATH, 'validated.tsv'), sep='\t')

#clips is the folder where we store our audio clips
clips_list = sorted(os.listdir(PATH + "/clips"))

def clip_validation_check(clip_name: str, text:list[str]) -> bool:
    """
    Checks if the given clip name exists in the text data.

    Params:
    clip_name (str): The name of the clip to check.
    text (list[str]): The list of text data to check against.

    Returns:
    bool: True if the clip name exists in the text data.
    """

    return clip_name in text['path'].values


def write_audio_and_txt(clip_name: str, transcript: str, path: str=PATH) -> str:
    """
    Writes the audio and text data for the given clip name into temp_input folder.

    Params:
    clip_name (str): The name of the clip to write data for.

    Returns:
    str: The id of the client.
    """
    target_audio, sampling_rate = sf.read(os.path.join(path, 'clips', clip_name))
    filename_no_extension = clip_name.split('.')[0]

    sf.write(os.path.join(path,f'temp_input/{filename_no_extension}.flac'), target_audio, sampling_rate)
    with open(os.path.join(path,f'temp_input/{filename_no_extension}.txt'), 'w') as f:
       f.write(transcript)
    return None



def align_input(path: str=PATH):
    '''
    Runs the Montreal Forced Aligner on the audio transcript (txt) to first generate a target dict.
    Then aligns the audio clip to the target dict to generate a TextGrid where the words and phonemes are aligned to the target dict.

    Intalled with::

       conda create -n aligner -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068

    '''
    command = ('conda run -n aligner mfa g2p'.split()
            + [os.path.join(path, "temp_input"), 'german_mfa',  os.path.join(path, "temp_input/target_dict.txt"), '--clean', '--overwrite'])
    subprocess.run(command)
    command = 'conda run -n aligner mfa configure -t'.split() + [os.path.join(path, "temp_output")]
    subprocess.run(command)
    command = ('conda run -n aligner mfa align'.split()
            + [os.path.join(path,"temp_input"),
                os.path.join(path,"temp_input/target_dict.txt"),
                'german_mfa',
                os.path.join(path,"temp_output"),
                '--clean'])
    subprocess.run(command)


# We extract our sampas
def extract_sampas_and_cut_audio(clip_name: str, id: str, path: str=PATH):
    """
    Extracts the sampas and cuts the audio word by word for the given clip name.

    Parameters:
    clip_name (str): The name of the clip to extract data from.
    id (str): The id of the client.

    Returns:
    list: A list of dictionaries of each word, each containing:
    'file_name' : name of the clip
    'label' : the spoken word
    'cp_norm' : normalized cp-trajectories
    'melspec_norm_recorded' : normalized mel spectrogram of the audio clip
    'melspec_norm_synthesized' : normalized mel spectrogram synthesized from the cp-trajectories
    'vector' : embedding vector of the word, based on fastText Embeddings
    'client_id' : id of the client
    """

    data = list()

    #remove extension for TextGrid
    filename_no_extension = clip_name.split('.')[0]
    target_audio, sampling_rate = sf.read(os.path.join(path, 'clips', clip_name))

    tg = textgrid.openTextgrid(os.path.join(path, "temp_output", filename_no_extension + ".TextGrid"), False)

    
    for word_index, word in enumerate(tg.getTier('words')):
        phones = list()
        phone_durations = list()
        for phone in tg.getTier('phones').entries:
            if phone.label == 'spn':
                break
            if phone.start >= word.end:
                break
            if phone.start < word.start:

                continue

            phones.append(sampa_convert_dict[phone.label])

            phone_durations.append(phone.end - phone.start)

        if not phones:
            continue

        # write seg file
        rows = []
        for i, phone in enumerate(phones):
            row = "name = %s; duration_s = %f;" % (phone, phone_durations[i])
            rows.append(row)
        text = "\n".join(rows)
        seg_file_name = str(os.path.join(path, f"temp_output/target_audio_word_{word_index}.seg"))
        with open(seg_file_name, "w") as text_file:
            text_file.write(text)

        # get tract files and gesture score
        seg_file_name = ctypes.c_char_p(seg_file_name.encode())

        ges_file_name = str(os.path.join(path, f"temp_output/target_audio_word_{word_index}.ges"))
        ges_file_name = ctypes.c_char_p(ges_file_name.encode())

        devnull = open('/dev/null', 'w')
        with contextlib.redirect_stdout(devnull):
            util.VTL.vtlSegmentSequenceToGesturalScore(seg_file_name, ges_file_name)
        tract_file_name = str(os.path.join(path, f"temp_output/target_audio_word_{word_index}.txt"))
        c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())

        util.VTL.vtlGesturalScoreToTractSequence(ges_file_name, c_tract_file_name)
        cps = util.read_cp(tract_file_name)
        
        
        cp_norm = util.normalize_cp(cps)
        wav_rec = target_audio[int(word.start * sampling_rate):int(word.end * sampling_rate)]
        melspec_norm_rec = util.normalize_mel_librosa(util.librosa_melspec(wav_rec, sampling_rate))
        wav_syn, wav_syn_sr = util.speak(cps)
        melspec_norm_syn = util.normalize_mel_librosa(util.librosa_melspec(wav_syn, wav_syn_sr))
        
        fasttext_vector = FASTTEXT_EMBEDDINGS.get_word_vector(word.label)

        melspec_norm_syn = util.pad_same_to_even_seq_length(melspec_norm_syn)

        data.append({'file_name':filename_no_extension, 'label':word.label, 'cp_norm':cp_norm, 
                     'mespec_norm_recorded':melspec_norm_rec, 'melspec_norm_synthesized': melspec_norm_syn
                     , 'vector':fasttext_vector, 'client_id':id})
        
    return data


def clips_to_df(path='.', df_name='', *, all_="continue"):
    """
    Takes as input a name for our dataframe, as well as the path to our mozilla common voice data and converts it into a DataFrame.


    Parameters:
    path (str): The path to our mozilla common voice dataset.
    all_ (str, None, pd.DataFrame) : 

    Returns:
    DataFrame where each word of our Dataset is stored as well as additionally in the rows:
    'file_name' : name of the clip
    'label' : the spoken word
    'cp_norm' : normalized cp-trajectories
    'melspec_norm_recorded' : normalized mel spectrogram of the audio clip
    'melspec_norm_synthesized' : normalized mel spectrogram synthesized from the cp-trajectories
    'vector' : embedding vector of the word, based on fastText Embeddings
    'client_id' : id of the client

    As well as a 'faulty_files.txt' to log all errors which occured.
    """
    if isinstance(all_, str) and all_ == "continue":
        all_ = pd.read_pickle(os.path.join(path, f'{df_name}.pkl'))
    elif all_ is None:
        all_ = pd.DataFrame()

    faulty_files = []
    #clips_list = os.listdir(os.path.join(path, 'clips'))
    clips_list = pd.read_csv(os.path.join(path, 'validated.tsv'), sep='\t')['path'].tolist()
    transcript_list = pd.read_csv(os.path.join(path, 'validated.tsv'), sep='\t')['sentence'].tolist()
    client_id_list = pd.read_csv(os.path.join(path, 'validated.tsv'), sep='\t')['client_id'].tolist()
    if os.path.exists(os.path.join(path, 'temp_input')):
        shutil.rmtree(os.path.join(path, 'temp_input'))
    if os.path.exists(os.path.join(path, 'temp_output')): 
        shutil.rmtree(os.path.join(path, 'temp_output'))


    for i in tqdm(range(len(clips_list))):
        try:
            if not os.path.exists(os.path.join(path, 'temp_input')):
                os.makedirs(os.path.join(path, 'temp_input'))


            write_audio_and_txt(clips_list[i], transcript_list[i], path)
            align_input(path)

            data = extract_sampas_and_cut_audio(clips_list[i], client_id_list[i], path) 

            df = pd.DataFrame(data)
            all_ = pd.concat([all_, df], ignore_index=True)

            # intermediate saving all_data to a pickle file
            all_.to_pickle(os.path.join(path, f'{df_name}.pkl'))

        except Exception as e:
            faulty_files.append(clips_list[i], str(e))
            print(f'an error occured with file {clips_list[i]}:{e}')

        finally:
            # Always run cleanup code, even if an error occurs
            if os.path.exists(os.path.join(path, 'temp_input')):
                shutil.rmtree(os.path.join(path, 'temp_input'))
            if os.path.exists(os.path.join(path, 'temp_output')): 
                shutil.rmtree(os.path.join(path, 'temp_output'))

    return all_


