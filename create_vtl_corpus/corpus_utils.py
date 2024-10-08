import os
import ctypes
import contextlib
import logging
import shutil
import re
import subprocess
import string
import random
from joblib import Parallel, delayed
import pandas as pd
import fasttext
import fasttext.util
from paule import util
from praatio import textgrid
import soundfile as sf
import librosa
import numpy as np



DICT = {
    "a": "a",
    "aj": "aI",
    "aw": "aU",
    "aː": "a:",
    "b": "b",
    "bʲ": "b",  # b' not possible with VTL
    "c": "k",  # trying k instead of c
    "cʰ": "s",  # c_h not possible in SAMPA but in X SAMPA, trying s instead of k to get a softer sound
    "d": "d",
    "dʒ": "dZ",
    "dʲ": "d",  # d' not possible with VTL
    "e": "e",
    "ej": "I",
    "f": "f",
    "fʲ": "f",  # f' not possible with VTL
    "h": "h",
    "i": "i",
    "iː": "i:",
    "j": "j",
    "k": "k",
    "kʰ": "k",  # "k_h", is not possible in SAMPA ibut in X SAMPA
    "l": "l",
    "m": "m",
    "mʲ": "m",  # m' not possible with VTL
    "m̩": "m",  # glottal stop is sadly also not avaible in VTL (seemingly)
    "n": "n",
    "n̩": "n",  # glottal stop is sadly also not avaible in VTL (seemingly)
    "o": "o",
    "ow": "aU",  # for some reason oU is not possible in VTL
    "p": "p",
    "pʰ": "p",  # "p_h", is not possible in SAMPA ibut in X SAMPA
    "pʲ": "p",  # p' not possible with VTL ( not explicitly tested but infered from other cases)
    "s": "s",
    "t": "t",
    "tʃ": "tS",
    "tʰ": "t",  # "t_h", is not possible in SAMPA ibut in X SAMPA
    "tʲ": "t",  # t' not possible with VTL ( not explicitly tested but infered from other cases)
    "u": "u",
    "uː": "u:",
    "v": "v",
    "vʲ": "v",  # v' not possible with VTL ( not explicitly tested but infered from other cases)
    "w": "U",
    "z": "z",
    "æ": "a",  # Near-open front unrounded vowel { not possible with VTL, replacing with open front unrounded vowel
    "ç": "C",
    "ð": "D",
    "ŋ": "N",
    "ɐ": "6",
    "ɑ": "o",  # open back unrounded vowel not possible with VTL, Close-mid back rounded vowel
    "ɑː": "o:",
    "ɒ": "O",  # acutal sampa Q
    "ɒː": "O",  # acutal  x sampa Q:
    "ɔ": "O",
    "ɔj": "OY",
    "ə": "@",
    "əw": "aU",  # @U not possible with VTL, this is a shaky mapping
    "ɚ": "@",  # @` not possible with VTL ( not explicitly tested but infered from other cases)
    "ɛ": "E",
    "ɛː": "E:",
    "ɜ": "2",  # Open-mid central unrounded vowel not possible with VTL, replacing Close-mid front rounded vowel since both sound kinda like the german ö
    "ɜː": "2:",
    "ɝ": "2",  # 2` not possible with VTL ( not explicitly tested but infered from other cases)
    "ɟ": "dZ",  # J not possible with VTL, its a rare phoneme and coudl either be approximated with j\ or dZ
    "ɡ": "g",
    "ɪ": "I",
    "ɫ": "l",  # ɫ not possible with VTL, this phoneme is a long l "dark l" and is not present in german( example world : "Allah" in Arabic)
    "ɫ̩": "l",  # glottal stop is sadly also not available in VTL (seemingly)
    "ɱ": "m",  # ɱ not possible with VTL, this phoneme is a labiodental nasal
    "ɲ": "n",  # ɲ not possible with VTL, this phoneme is a palatal nasal, kind of like the spanish ñ
    "ɹ": "r",
    "ɾ": "r",  # ɾ ( SAMPA : 4) not possible with VTL, this phoneme is a alveolar tap
    "ʃ": "S",
    "ʉ": "u",  # ʉ not possible with VTL(X-SAMPA : } ), this phoneme is a close central rounded vowel
    "ʉː": "u:",
    "ʊ": "U",
    "ʎ": "l",  # ʎ not possible with VTL (SAMPA L ), this phoneme is a palatal lateral approximant
    "ʒ": "Z",
    "ʔ": "?",  # basic glottal stop, alone prduces no sound in VTL
    "θ": "T",
    "ʁ": "R",
    "eː": "e:",
    "x": "x",
    "ts": "ts",
    "ɔʏ": "OY",
    "oː": "o:",
    "œ": "9",
    "yː": "y:",
    "ʏ": "Y",
    "øː": "2:",
    "ø": "2",
    "pf": "pf",
    "l̩": "l",  # glottal stop is sadly also not avaible in VTL (seemingly) at least as an additive to a phoneme
    "t̪": "T",  #   t̪ not possible with VTL (SAMPA t_d) ( not explicitly tested but infered from other cases)
    "ʈʲ": "T",  # t` not possible with VTL ( not explicitly tested but infered from other cases)
    "ʈ": "t",  # t` not possible with VTL ( not explicitly tested but infered from other cases)
    "ʋ": "v",  #    X-SAMPA P or v\ according to Wikipedia, but not available in VTL
    "d̪": "d",  # d_d not possible with VTL ( and maybe not correct phoneme as well)
    "kʷ": "k",  # k_w not possible with VTL ( and maybe not correct phoneme as well)
    "cʷ": "C",  # c_w not possible with VTL ( and maybe not correct phoneme as well)
    "ɖ": "d",  # d` not possible with VTL ( and maybe not correct phoneme as well)
    "tʷ": "t",  # t_w not possible with VTL (inferring from other cases)
    "ɟʷ": "dZ",  # J_w not possible with VTL (inferring from other cases)
}  # this dict can be made shorter with : automatically passing etc



def generate_rows(filename_no_extension,sentence, path_to_corpus):
            """This function is used to create the matching rows from a clip
            It is used for the multiprocessing part of the code
            Parameters: filename_no_extension (str): The name of the clip
                        sentence (str): The sentence of the clip

            Returns: The rows for the dataframe as a dataframe object  (pd.DataFrame)"""
            
            
           

            labels = list()
            word_positions = list()
            sentences = list()
            wavs = list()
            wavs_sythesized = list()
            sampling_rates = list()
            sampling_rates_sythesized = list()
            phone_durations_list = list()
            sampa_phones = list()
            cp_norms = list()
            melspecs_norm_recorded = list()
            melspecs_norm_synthesized = list()
            vectors = list()
            client_ids = list()
            names = list()
            mfa_phones = list()
            lexical_words = list()

            clip_name = filename_no_extension + ".mp3"

            target_audio, sampling_rate = sf.read(
                os.path.join(path_to_corpus, "clips_validated", clip_name)
            )

            assert (
                len(target_audio.shape) == 1
            ), f"The audio file {clip_name} is not mono"
            try:
                tg = textgrid.openTextgrid(
                    os.path.join(
                        path_to_corpus + "_aligned", filename_no_extension + ".TextGrid"
                    ),
                    False,
                )
            except FileNotFoundError:
                logging.warning(
                    f"The TextGrid file for {filename_no_extension} was not found"
                )
                clip_list.remove(filename_no_extension)
                sentence_list.remove(sentence)
                files_skiped += 1
                return df_part
                
            text_grid_sentence = list()
            
            for word_index, word in enumerate(tg.getTier("words")):
                text_grid_sentence.append(word.label)
            logging.info(sentence)
            logging.info(text_grid_sentence)
            for word_index, word in enumerate(tg.getTier("words")):

                phones = list()
                mfa_phones_word_level = list()

                phone_durations = list()
                for phone in tg.getTier("phones").entries:
                    if phone.label == "spn": 
                        break
                    if phone.start >= word.end:
                        break
                    if phone.start < word.start:

                        continue

                    mfa_phone = phone.label
                    mfa_phones_word_level.append(mfa_phone)
                    sampa_phone = DICT[mfa_phone]
                    phones.append(sampa_phone)
                    mfa_phones_word_level.append(mfa_phone)

                    phone_durations.append(phone.end - phone.start)

                if not phones:
                    logging.warning(
                        f"No phones found for word '{word.label}' in {filename_no_extension}, skipping this word"
                    )
                    continue
                logging.info(
                    f"Processing word '{word.label}' in {filename_no_extension}, resulting phones: {phones}"
                )
                # splicing audio
                wav_rec = target_audio[
                    int(word.start * sampling_rate) : int(word.end * sampling_rate)
                ]
                assert wav_rec is not None, "The audio is None"
                split_sentence = re.split(r"[ -]|\.\.", sentence)
                maximum_word_index = len(split_sentence) - 1
                if word_index > maximum_word_index:
                    logging.warning(
                        f"Word index {word_index} is greater than the maximum index {maximum_word_index} of the sentence in {filename_no_extension}, skipping this word, Sentence: {sentence} .last word: {sentence.split()[-1]}"
                    )
                
                lexical_word = (
                    split_sentence[word_index]
                    .replace(".", "")
                    .replace(",", "")
                    .replace("?", "")
                    .replace("!", "")
                    .replace(":", "")
                    .replace(";", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace('"', "")
                    .replace("'", "")
                )  # remove dots ( might impact pronunciation however)
                lexical_words.append(lexical_word)
                assert (
                    word.label.lower().replace("'", "") == lexical_word.lower()
                ), f"Word mismatch since '{word.label.lower() .replace("'", "")}' is not equal to '{lexical_word.lower()} in sentence '{sentence}' in {filename_no_extension}. TextGrid sentece: {text_grid_sentence}"
                
                labels = list()
                word_positions = list()
                sentences = list()
                wavs = list()
                wavs_sythesized = list()
                sampling_rates = list()
                sampling_rates_sythesized = list()
                phone_durations_list = list()
                sampa_phones = list()
                cp_norms = list()
                melspecs_norm_recorded = list()
                melspecs_norm_synthesized = list()
                vectors = list()
                client_ids = list()
                names = list()
                mfa_phones = list()
                lexical_words = list()
                

                #random id 
                client_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

                logging.info(f"Client id: {client_id}, writing seg file")
                # write seg file
                rows = []
                for i, phone in enumerate(phones):
                    row = "name = %s; duration_s = %f;" % (phone, phone_durations[i])
                    rows.append(row)
                text = "\n".join(rows)
                path = os.path.join(path_to_corpus, "clips")
                if not os.path.exists(path):
                    os.mkdir(path=path)
                # delete this later?
                if not os.path.exists(os.path.join(path, "temp_output")):
                    os.mkdir(path=os.path.join(path, "temp_output"))
                seg_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}_{client_id}.seg"
                    )
                )
                if os.path.exists(seg_file_name):
                    client_id2 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                    logging.warning( f"Client id: {client_id} already exists, creating new one: {client_id2}. Consider deleting the temp_output folder")
                    client_id = client_id2
                    seg_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}_{client_id}.seg"
                    )
                )
                with open(seg_file_name, "w") as text_file:
                    text_file.write(text)

                # get tract files and gesture score
                seg_file_name = ctypes.c_char_p(seg_file_name.encode())

                ges_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}_{client_id}.ges"
                    )
                )
                ges_file_name = ctypes.c_char_p(ges_file_name.encode())
                logging.debug(f"Client id: {client_id}, writing ges file for word: {word.label}")
                devnull = open("/dev/null", "w")
                with contextlib.redirect_stdout(devnull):
                    util.VTL.vtlSegmentSequenceToGesturalScore(
                        seg_file_name, ges_file_name
                    )
                tract_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}_{client_id}.txt"
                    )
                )
                c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())

                logging.debug(f"Client id: {client_id}, writing tract file, for word: {word.label}")

                util.VTL.vtlGesturalScoreToTractSequence(
                    ges_file_name, c_tract_file_name
                )
                cps = util.read_cp(tract_file_name)
                logging.debug(f"Client id: {client_id}, normalizing cps for word: {word.label}")
                cp_norm = util.normalize_cp(cps)
                cp_norms.append(cp_norm)
                logging.debug(f"Client id: {client_id}, writing melspec for word: {word.label}")
                

                # resample and extract melspec but it needs to be skipped for now since it is hanging the process
                """
                logging.info(f"Client id: {client_id}, commencing resampling for word: {word.label} (Andres bet)")
                wav = librosa.resample(wav_rec, orig_sr=sampling_rate, target_sr=44100,
                            res_type='kaiser_best', fix=True, scale=False)
                logging.info(f"Client id: {client_id}, commencing feature extraction for word: {word.label} (Tinos bet)")
                melspec = librosa.feature.melspectrogram(y=wav, n_fft=1024, hop_length=220, n_mels=60, sr=44100, power=1.0, fmin=10, fmax=12000)
                logging.info(f"Client id: {client_id}, commencing amplitdue thing for word: {word.label} ")
                melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)
                logging.info(f"Client id: {client_id}, converting melspec for word: {word.label}")
                melspec_rec = np.array(melspec_db.T, order='C', dtype=np.float64)


                
                logging.info(f"Client id: {client_id}, normalizing melspec for word: {word.label}")
                melspec_norm_rec = util.normalize_mel_librosa(
                   melspec_rec
                )
                loggin.info(f"Completed melspec for word: {word.label}, client_id: {client_id}")

                melspecs_norm_recorded.append(melspec_norm_rec)
                """

                melspecs_norm_recorded.append(None)
                logging.debug(f"Starting synthesis for {word.label} on  client_id {client_id}")
                wav_syn, wav_syn_sr = util.speak(cps)
                wavs_sythesized.append(wav_syn)
                sampling_rates_sythesized.append(wav_syn_sr)

                """
                melspec_norm_syn = util.normalize_mel_librosa(
                    util.librosa_melspec(wav_syn, wav_syn_sr)
                )

                melspec_norm_syn = util.pad_same_to_even_seq_length(melspec_norm_syn)
                melspecs_norm_synthesized.append(melspec_norm_syn)
                """

                melspecs_norm_synthesized.append(None)
                if len(names) != len(wavs):
                    print(
                        f"The wavs are not the same length,at '{word.label}' Expected: {len(names)}) but got {len(wavs)}"
                    )
            # fill the dataframe

            df_part = pd.DataFrame(
            {
                "file_name": names,
                "label": labels,
                "lexical_word": lexical_words,
                "word_position": word_positions,
                "sentence": sentences,
                "wav_recording": wavs,
                "wav_synthesized": wavs_sythesized,
                "sr_recording": sampling_rates,
                "sr_synthesized": sampling_rates_sythesized,
                "sampa_phones": sampa_phones,
                "mfa_phones": mfa_phones,
                "phone_durations": phone_durations_list,
                "cp_norm": cp_norms,
                "melspec_norm_recorded": melspecs_norm_recorded,
                "melspec_norm_synthesized": melspecs_norm_synthesized,
                "vector": vectors,
                "client_id": client_ids,
            })
            return df_part
