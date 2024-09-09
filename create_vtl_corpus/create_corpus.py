import os
import argparse
import ctypes
import contextlib
import logging
import shutil
import re
import subprocess
from joblib import Parallel, delayed
import pandas as pd
import fasttext
import fasttext.util
from paule import util
from praatio import textgrid
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor  
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DIR = os.path.dirname(__file__)


class CreateCorpus:
    """
    This class generates the vocaltract lab trajectories for a corpus.
    It's assumed that a corpus has the following shape
    as is common with Mozillas Common Voice Corpus using MFA

    corpus/
    ├── validated.txt          a file where the transripts are stored
    |
    ├── clips/
    |     └──*.mp3
    └── *.files_not_relevant_to this project

    Attributes:
    ---------------
    path_to_corpus: str
        The path to the corpus
    language: str
        The language of the corpus as an abbreviation

    Methods:
    ---------------
    format_corpus():
        Takes the path to the corpus and formats it to the fitting form
    run_aligner():
        Runs the Montreal Forced Aligner on the corpus
    check_structure():
        Checks if the corpus has the right format and if not corrects this
    create_dataframe():
        Extracts the SAMPA phonemes from the aligned corpus
    """

    fast_text_model_english = "cc.en.300.bin"
    fast_text_model_german = "cc.de.300.bin"

    @staticmethod
    def setup(language: str):

        if language == "en":
            model_name = CreateCorpus.fast_text_model_english

        elif language == "de":
            model_name = CreateCorpus.fast_text_model_german
        else:
            raise ValueError("The language is not supported")
        path_to_text_model = os.path.join(DIR, "resources", model_name)
        if not os.path.exists(path_to_text_model):
            logging.warning("FastTextModel doesn't exist. Downloading fasttext model")
            fasttext.util.download_model(language, if_exists="ignore")
            shutil.move(f"cc.{language}.300.bin", path_to_text_model)
            os.remove(f"cc.{language}.300.bin.gz")

    def __init__(self, path_to_corpus: str, *, language: str):
        self.path_to_corpus = path_to_corpus
        self.language = language
        self.fast_text_model = self.load_fasttext_model(language)
        self.mfa_to_sampa_dict = {
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

    def load_fasttext_model(self, language: str):
        """
        Loads the fasttext model for the given language

        Params:
        language (str): The language of the model

        Returns:
        fasttext.FastText._FastText: The loaded fasttext model
        """
        if language == "en":
            model = fasttext.load_model(
                os.path.join(DIR, "resources", CreateCorpus.fast_text_model_english)
            )
        elif language == "de":
            model = fasttext.load_model(
                os.path.join(DIR, "resources", CreateCorpus.fast_text_model_german)
            )
        else:
            raise ValueError("The language is not supported")
        logging.info("Fasttext model loaded")
        return model

    def format_corpus(self):
        """
        Takes the path to the corpus and formats it to the fitting form

        Params:
        -

        Returns:
        -
        """

        data = pd.read_table(
            os.path.join(self.path_to_corpus, "validated.tsv"), sep="\t"
        )
        clip_names = list()
        sentences = list()
        senteces_that_are_not_strings = 0
        if not os.path.exists(os.path.join(self.path_to_corpus, "clips_validated")):
            os.mkdir(os.path.join(self.path_to_corpus, "clips_validated"))
            need_new_clips = True
            logging.info(
                "The clips_validated folder was created, validated clips will be copied from the clips folder"
            )
        else:
            logging.info(
                "The clips_validated folder already exists. We assume it is filled already. If not delete and rerun"
            )
            need_new_clips = False

        for _, row in data.iterrows():
            transcript = row["sentence"]
            clip_name = row["path"].removesuffix(".mp3")
            clip_names.append(clip_name)
            sentences.append(transcript)
            if need_new_clips:
                shutil.copy(
                    os.path.join(self.path_to_corpus, "clips", clip_name + ".mp3"),
                    os.path.join(
                        self.path_to_corpus, "clips_validated", clip_name + ".mp3"
                    ),
                )

                file_name = clip_name + ".lab"
                with open(
                    os.path.join(self.path_to_corpus, "clips_validated", file_name),
                    "wt",
                ) as lab_file:
                    # print(transcript)
                    if not isinstance(transcript, str):
                        logging.debug(
                            f"Transcript for {clip_name} is not a string, skipping this clip"
                        )
                        senteces_that_are_not_strings += 1
                        continue
                    lab_file.write(transcript)

        if senteces_that_are_not_strings > 0:
            logging.warning(
                f"{senteces_that_are_not_strings} sentences were not strings and were skipped. Thats {senteces_that_are_not_strings/len(clip_names)*100}% of the sentences"
            )
        else:
            logging.info("All sentences were strings")
        return clip_names, sentences

    def run_aligner(self, mfa_workers: int, batch_size: int):
        """
                 Runs the Montreal Forced Aligner on the corpus
                Params:
                mfaworkers (int): The number of workers to use

                Returns:
                -
        """
        path_to_validated = os.path.join(self.path_to_corpus, "clips_validated")
        path_to_aligned = os.path.join(self.path_to_corpus,  "clips_aligned")
        files_in_validated = list()
        for file in os.listdir(path_to_validated):
            if file.endswith(".mp3"):
                files_in_validated.append(os.path.splitext(file)[0])
        lenght_files = len(files_in_validated)
        logging.info(f"Number of files in validated: {files_in_validated}")
        if lenght_files >= batch_size:
            num_batches = (lenght_files + batch_size -1) // batch_size
            logging.warning(f"The number of files is over f{batch_size}, so we split the aligner into {num_batches} batches")
            logging.info(f"we now need to split the validated folder into {num_batches} subdirectories")

            # Move files in batches
            logging.info(f"Moving files in batches of {batch_size} and aligning them")
            for batch_number in tqdm(range(num_batches)):
                # Create a subdirectory for the current batch
                batch_folder = os.path.join(path_to_validated, f'batch_{batch_number+1}')
                os.makedirs(batch_folder, exist_ok=True)

                # Determine the range of files to move for the current batch
                start_index = batch_number * batch_size
                end_index = start_index + batch_size
                batch_files = files_in_validated[start_index:end_index]

                # Move each file in the current batch
                for i, file in enumerate(batch_files):
                    mp3_file =  file + ".mp3"
                    source_path = os.path.join(path_to_validated, mp3_file)
                    batch_path = os.path.join(batch_folder, mp3_file)
                    shutil.copy(source_path, batch_path) # Copy the file to the batch folder as a safety measure, could be changed to move
                    lab_file = file + ".lab"
                    if not os.path.exists(os.path.join(path_to_validated, lab_file)):
                        logging.warning(f"Lab file {lab_file} does not exist, skipping this file")
                        continue
                    else:
                        source_path = os.path.join(path_to_validated, lab_file)
                        batch_path = os.path.join(batch_folder, lab_file)
                        shutil.copy(source_path, batch_path) # Copy the file to the batch folder as a safety measure, could be changed to move

                   
                logging.info(f"Moved {len(batch_files)} files to {batch_folder}")

                if self.language == "en":
                    logging.info("aligning corpus in english")
                    command = "conda run -n aligner mfa  align".split() + [
                    batch_folder,
                        "english_mfa",
                        "english_mfa",
                    path_to_aligned,
                        f"--num_jobs {mfa_workers}",
                        "--use_mp",
                    ]

                if self.language == "de":
                    logging.info("aligning corpus in german")
                    command = "conda run -n aligner mfa  align".split() + [
                    batch_folder,
                        "german_mfa",
                        "german_mfa",
                    path_to_aligned,
                        f"--num_jobs {mfa_workers}",
                        "--use_mp",
                    ]

                run = subprocess.run(command)
                assert run.returncode == 0, f"The aligner did not run successfully for batch {batch_number +1} that was run"
                logging.info(f"Batch {batch_number +1} was aligned successfully")
                shutil.rmtree(batch_folder) # once the batch is done, remove the folder only makes sense in the case of copying
                logging.info(f"Removed {batch_folder}, to save space")

              
            



        else: 
            if self.language == "en":
                logging.info("aligning corpus in english")
                command = "conda run -n aligner mfa  align".split() + [
                    path_to_validated,
                    "english_mfa",
                    "english_mfa",
                    path_to_aligned,
                    f"--num_jobs {mfa_workers}",
                    "--use_mp",
                ]

            if self.language == "de":
                logging.info("aligning corpus in german")
                command = "conda run -n aligner mfa  align".split() + [
                path_to_validated,
                    "german_mfa",
                    "german_mfa",
                path_to_aligned,
                    f"--num_jobs {mfa_workers}",
                    "--use_mp",
                ]

            run = subprocess.run(command)
            assert run.returncode == 0, "The aligner did not run successfully for the single batch that was run"
        logging.info("The aligner ran successfully")
    def check_structure(self):
        """
        Checks if the corpus has the right  and if not corrects this

        Params:
        -

        Returns:
         clipnames: List[str]
            A set of the clip names
        """
        assert os.path.exists(
            os.path.join(self.path_to_corpus, "validated.tsv")
        ), "No validated.tsv found"
        with_clips = os.path.join(self.path_to_corpus, "clips")
        assert os.path.exists(with_clips), "No clips folder found"
        lab_files = set()
        mp3_files = set()

        # Traverse the directory and collect the file names
        for file in os.listdir(with_clips):
            if file.endswith(".lab"):

                lab_files.add(os.path.splitext(file)[0])
            elif file.endswith(".mp3"):
                mp3_files.add(os.path.splitext(file)[0])

        if mp3_files == lab_files:

            clip_names = lab_files
            logging.info("The lab files and mp3 files match")
            # TODO: Implent this correctly
        ## Find a way to use the sentences from the validated.tsv file
        else:
            logging.warning(
                "The lab files and mp3 files do not match, correcting this now"
            )
        clip_names, sentence_list = self.format_corpus()
        missing_clips = set(clip_names).difference(lab_files.intersection(mp3_files))
        assert (
            missing_clips == set()
        ), f"The lab files and mp3 files do not match, since the following clips are missing: {missing_clips}"
        return clip_names, sentence_list

    def create_data_frame(
        self, path_to_corpus: str, clip_list: list, sentence_list: list, num_cores: int
    ):
        """
        Creates Dataframe with Vocaltract Lab data and other data
        Parameters:
        path_to_corpus (str): The path to the corpus
        clip_list (list): A list of the clip names present in the corpus

        Returns:
        Dataframe: A dataframe with the following labels
        'file_name' : name of the clip
        'label' : the spoken wordn
        'lexical_word' : the word as it is in the dictionary
        'word_position' : the position of the word in the sentence
        'sentence' : the sentence the word is part of
        'wav_recording' : spliced out audio as mono audio signal
        'sr_recording' : sampling rate of the recording
        'sampa_phones' : the sampa(like) phonemes of the word
        "mfa_phones" : the phonemes as outputted by the aligner
        'phone_durations_lists' : the duration of each phone in the word as list
        'cp_norm' : normalized cp-trajectories
        'melspec_norm_recorded' : normalized mel spectrogram of the audio clip
        'melspec_norm_synthesized' : normalized mel spectrogram synthesized from the cp-trajectories
        'vector' : embedding vector of the word, based on fastText Embeddings
        'client_id' : id of the client

        """

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

        used_phonemes = set()
        files_skiped = 0
        index = 0
        # remove extension for TextGrid
             
        def return_row_from_clip(filename_no_extension,sentence):
            """This function is used to create the matching row from a clip
            It is used for the multiprocessing part of the code
            Parameters: filename_no_extension (str): The name of the clip
                        sentence (str): The sentence of the clip
            Returns: The row for the dataframe as a list"""
            pass 
            """
            df_row = list()
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
                continue
                
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
                    sampa_phone = self.mfa_to_sampa_dict[mfa_phone]
                    phones.append(sampa_phone)
                    mfa_phones_word_level.append(mfa_phone)

                    phone_durations.append(phone.end - phone.start)

                if not phones:
                    pass
                logging.debug(
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
                df_row.append(clip_name)
                df_row.append(word.label)
                df_row.append(word_index)
                df_row.append(phones)
                df_row.append(phone_durations)
                df_row.append(wav_rec)
                df_row.append(sampling_rate)
                df_row.append(mfa_phones_word_level)
                df_row.append(lexical_word)
                df_row.append(sentence)
                

                # write seg file
                rows = []
                for i, phone in enumerate(phones):
                    row = "name = %s; duration_s = %f;" % (phone, phone_durations[i])
                    rows.append(row)
                text = "\n".join(rows)
                path = os.path.join(path_to_corpus + "_aligned", "clips")
                if not os.path.exists(path):
                    os.mkdir(path=path)
                # delete this later?
                if not os.path.exists(os.path.join(path, "temp_output")):
                    os.mkdir(path=os.path.join(path, "temp_output"))
                seg_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}.seg"
                    )
                )
                with open(seg_file_name, "w") as text_file:
                    text_file.write(text)

                # get tract files and gesture score
                seg_file_name = ctypes.c_char_p(seg_file_name.encode())

                ges_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}.ges"
                    )
                )
                ges_file_name = ctypes.c_char_p(ges_file_name.encode())

                devnull = open("/dev/null", "w")
                with contextlib.redirect_stdout(devnull):
                    util.VTL.vtlSegmentSequenceToGesturalScore(
                        seg_file_name, ges_file_name
                    )
                tract_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}.txt"
                    )
                )
                c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())

                util.VTL.vtlGesturalScoreToTractSequence(
                    ges_file_name, c_tract_file_name
                )
                cps = util.read_cp(tract_file_name)

                cp_norm = util.normalize_cp(cps)
                cp_norms.append(cp_norm)

                melspec_norm_rec = util.normalize_mel_librosa(
                    util.librosa_melspec(wav_rec, sampling_rate)
                )

                melspecs_norm_recorded.append(melspec_norm_rec)
                wav_syn, wav_syn_sr = util.speak(cps)
                wavs_sythesized.append(wav_syn)
                sampling_rates_sythesized.append(wav_syn_sr)
                melspec_norm_syn = util.normalize_mel_librosa(
                    util.librosa_melspec(wav_syn, wav_syn_sr)
                )

                melspec_norm_syn = util.pad_same_to_even_seq_length(melspec_norm_syn)
                melspecs_norm_synthesized.append(melspec_norm_syn)
                if word.label == "chocolate":
                    sf.write("manual_tests/chocolate.wav", wav_rec, sampling_rate)
                    import matplotlib.pyplot as plt

                    util.librosa.display.specshow(melspec_norm_rec, x_axis="time")
                    plt.colorbar()
                    plt.savefig("manual_tests/chocolate.png")
                    with open(
                        "manual_tests/chocolate_updated_again.seg", "w"
                    ) as text_file:
                        text_file.write(text)

                if len(names) != len(wavs):
                    print(
                        f"The wavs are not the same length,at '{word.label}' Expected: {len(names)}) but got {len(wavs)}"
                    )
                if word.label == "utopie":
                    sf.write("manual_tests/Utopie.wav", wav_rec, sampling_rate)
                    import matplotlib.pyplot as plt

                    util.librosa.display.specshow(melspec_norm_rec, x_axis="time")
                    plt.colorbar()
                    plt.savefig("manual_tests/Utopie.png")
                    with open("manual_tests/Utopie.seg", "w") as text_file:
                        text_file.write(text)

        # with Pool(n_jobs) as pool:
        #    list_of_rows = pool.map(create_rows_from_clip, clip_list)
        """
        path_to_aligned = os.path.join(self.path_to_corpus,  "clips_aligned")
        for filename_no_extension, sentence in tqdm(
            zip(clip_list, sentence_list), total=len(clip_list)
        ):
            
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
                      path_to_aligned, filename_no_extension + ".TextGrid"
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
                continue
                
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
                    sampa_phone = self.mfa_to_sampa_dict[mfa_phone]
                    used_phonemes.add(sampa_phone)
                    phones.append(sampa_phone)
                    mfa_phones_word_level.append(mfa_phone)

                    phone_durations.append(phone.end - phone.start)

                if not phones:
                    continue
                logging.debug(
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
                names.append(clip_name)
                sampa_phones.append(phones)
                phone_durations_list.append(phone_durations)
                mfa_phones.append(mfa_phones_word_level)
                wavs.append(wav_rec)
                # adding easy to add variables to the lists
                labels.append(word.label)
                sampling_rates.append(sampling_rate)
                word_positions.append(word_index)
                fasttext_vector = self.fast_text_model.get_word_vector(word.label)
                vectors.append(fasttext_vector)
                client_ids.append(filename_no_extension)
                sentences.append(sentence)

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
                        path, f"temp_output/target_audio_word_{word_index}.seg"
                    )
                )
                with open(seg_file_name, "w") as text_file:
                    text_file.write(text)

                # get tract files and gesture score
                seg_file_name = ctypes.c_char_p(seg_file_name.encode())

                ges_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}.ges"
                    )
                )
                ges_file_name = ctypes.c_char_p(ges_file_name.encode())

                devnull = open("/dev/null", "w")
                with contextlib.redirect_stdout(devnull):
                    util.VTL.vtlSegmentSequenceToGesturalScore(
                        seg_file_name, ges_file_name
                    )
                tract_file_name = str(
                    os.path.join(
                        path, f"temp_output/target_audio_word_{word_index}.txt"
                    )
                )
                c_tract_file_name = ctypes.c_char_p(tract_file_name.encode())

                util.VTL.vtlGesturalScoreToTractSequence(
                    ges_file_name, c_tract_file_name
                )
                cps = util.read_cp(tract_file_name)

                cp_norm = util.normalize_cp(cps)
                cp_norms.append(cp_norm)

                melspec_norm_rec = util.normalize_mel_librosa(
                    util.librosa_melspec(wav_rec, sampling_rate)
                )

                melspecs_norm_recorded.append(melspec_norm_rec)
                wav_syn, wav_syn_sr = util.speak(cps)
                wavs_sythesized.append(wav_syn)
                sampling_rates_sythesized.append(wav_syn_sr)
                melspec_norm_syn = util.normalize_mel_librosa(
                    util.librosa_melspec(wav_syn, wav_syn_sr)
                )

                melspec_norm_syn = util.pad_same_to_even_seq_length(melspec_norm_syn)
                melspecs_norm_synthesized.append(melspec_norm_syn)
                if word.label == "chocolate":
                    sf.write("manual_tests/chocolate.wav", wav_rec, sampling_rate)
                    import matplotlib.pyplot as plt

                    util.librosa.display.specshow(melspec_norm_rec, x_axis="time")
                    plt.colorbar()
                    plt.savefig("manual_tests/chocolate.png")
                    with open(
                        "manual_tests/chocolate_updated_again.seg", "w"
                    ) as text_file:
                        text_file.write(text)

                if len(names) != len(wavs):
                    print(
                        f"The wavs are not the same length,at '{word.label}' Expected: {len(names)}) but got {len(wavs)}"
                    )
                if word.label == "utopie":
                    sf.write("manual_tests/Utopie.wav", wav_rec, sampling_rate)
                    import matplotlib.pyplot as plt

                    util.librosa.display.specshow(melspec_norm_rec, x_axis="time")
                    plt.colorbar()
                    plt.savefig("manual_tests/Utopie.png")
                    with open("manual_tests/Utopie.seg", "w") as text_file:
                        text_file.write(text)

        logging.info(f"Used phonemes: {used_phonemes}")
        for idx, array in enumerate(
            [
                names,
                labels,
                word_positions,
                sentences,
                wavs,
                sampling_rates,
                sampa_phones,
                phone_durations_list,
                cp_norms,
                melspecs_norm_recorded,
                melspecs_norm_synthesized,
                vectors,
                client_ids,
                mfa_phones,
                lexical_words,
            ]
        ):
            logging.info(f"Length of array {idx}: {len(array)}")

        df = pd.DataFrame(
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
            }
        )
        shutil.rmtree(
            os.path.join(path_to_corpus + "_aligned" + "/clips/temp_output")
        )  # we don't need the temp_output files anymore
        logging.info(f"Files skipped: {files_skiped}")
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a corpus to the vocaltract lab format"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="../../cv-corpus-18.0-delta-2024-06-14/de",
        help="The path to the corpus which should be converted to the vocaltract lab format",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="The language of the corpus as an abbreviation",
    )
    parser.add_argument(
        "--mfa_workers",
        type=int,
        default=6,
        help="The number of mfa workers to use",
    )
    parser.add_argument(
        "--needs_aligner",
        action="store_true",
        default=False,
        help="If the aligner should be run",
    )
    parser.add_argument(
        "--search_df",
        action="store_true",
        default=False,
        help=" If a already created dataframe should be searched for and then used instead of creating a new one",
    )
    parser.add_argument(
        "--df_path", type=str, default=None, help="The path to the dataframe"
    )  # TODO

    parser.add_argument(
        "--aligner_batch_size", type=int, default=5000, help="How many text files the aligner should process in one batch")

    parser.add_argument(
        "--num_cores", type=int, default=-1, help="The number of jobs the aligner should use, uses maximum on default")
    args = parser.parse_args()

    assert os.path.isdir(args.corpus), "The provided path is not a directory"
    CreateCorpus.setup(language=args.language)
    corpus_worker = CreateCorpus(args.corpus, language=args.language)
    if args.search_df:
        pass
    clip_list, sentence_list = corpus_worker.check_structure()
    if args.needs_aligner:
        mfa_workers = args.mfa_workers
        corpus_worker.run_aligner(mfa_workers, args.aligner_batch_size)
    df = corpus_worker.create_data_frame(args.corpus, clip_list, sentence_list, args.num_cores)
    logging.info(df)
    path_to_save_corpus = os.path.join(args.corpus, "corpus_as_df.pkl")
    df.to_pickle(path_to_save_corpus)

    logging.info("Done! :P")
