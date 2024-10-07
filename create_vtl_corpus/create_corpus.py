import os
import argparse
import ctypes
import contextlib
import logging
import shutil
import re
import time
import subprocess
from joblib import Parallel, delayed
import pandas as pd
import fasttext
import fasttext.util
from paule import util
from praatio import textgrid
import soundfile as sf
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from corpus_utils import (
    generate_rows,
    DICT,
    FASTTEXT_EN,
    FASTTEXT_DE,
    replace_special_chars
)

# Set up logging
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
        self.mfa_to_sampa_dict = DICT

    def load_fasttext_model(self, language: str):
        """
        Loads the fasttext model for the given language

        Params:
        language (str): The language of the model

        Returns:
        fasttext.FastText._FastText: The loaded fasttext model
        """
        if language == "en":
            global FASTTEXT_EN
            FASTTEXT_EN = fasttext.load_model(
                os.path.join(DIR, "resources", "cc.en.300.bin")
            )
            model = FASTTEXT_EN
        elif language == "de":
            global FASTTEXT_DE
            FASTTEXT_DE = fasttext.load_model(
                os.path.join(DIR, "resources", "cc.de.300.bin")
            )
            model = FASTTEXT_DE

        else:
            raise ValueError("The language is not supported")
        logging.info("Fasttext model loaded")
        return model

    def format_corpus(self, word_amount, min_word_count):
        """
        Takes the path to the corpus and formats it to the fitting form

        Params:
        word_amount (int): The amount of words that should be used, if 0 all clips are used
        word_a

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

        assert word_amount >= 0, "Amount of Sentences cannot be negative"
        assert min_word_count >= 0, "Minimum word count cannot be negative"
        if word_amount != 0:
            assert (
                word_amount >= min_word_count 
            ), "The word amount cannot be smaller than the minimum word count"
        self.word_amount = word_amount
        self.min_word_count = min_word_count

        self.create_frozen_set(data["sentence"], word_amount, min_word_count)

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

    def create_frozen_set(self, validated_sentences, word_amount, min_word_count):
        """
        Creates a frozen set of the words that should be used in the corpus and saves it as a class attribute
        Params:
        validated_sentences (pd.Series): The sentences from the validated.tsv file
        word_amount (int): The amount of words that should be used, if 0 all clips are used
        min_word_count (int): The minimum amount of words a word should have to be included in  the word amount argument
        Returns:
        -
        """
        all_words = validated_sentences.str.split().explode().dropna()

        word_counts = Counter(all_words)


        logging.info(f"{word_counts} These are the word counts")
        filtered_word_counts = {
            word: count
            for word, count in word_counts.items()
            if count >= min_word_count
        }
        
        if word_amount > 0:
            word_set = frozenset(
                key.lower()
                for key, _ in list(filtered_word_counts.items())[:word_amount]
                if isinstance(key, str)
            )
        else:
            word_set = frozenset(
                key.lower()
                for key, _ in filtered_word_counts.items()
                if isinstance(key, str)
            )

        assert len(word_set) > 0, "The word set is empty, no words were found"
        assert (
            len(word_set) == word_amount or word_amount == 0
        ), f"The word set has {len(word_set)} words, but the word amount is {word_amount}"
        self.word_set = word_set

        logging.info(f"{word_set} These are the words that will be used in the corpus")

    def run_aligner(self, mfa_workers: int, batch_size: int):
        """
            Runs the Montreal Forced Aligner on the corpus
           Params:
           mfaworkers (int): The number of workers to use

           Returns:
        -
        """
        path_to_validated = os.path.join(self.path_to_corpus, "clips_validated")
        path_to_aligned = os.path.join(self.path_to_corpus, "clips_aligned")
        files_in_validated = list()
        for file in os.listdir(path_to_validated):
            if file.endswith(".mp3"):
                files_in_validated.append(os.path.splitext(file)[0])
        lenght_files = len(files_in_validated)
        logging.info(f"Number of files in validated: {files_in_validated}")
        if lenght_files >= batch_size:
            num_batches = (lenght_files + batch_size - 1) // batch_size
            logging.warning(
                f"The number of files is over f{batch_size}, so we split the aligner into {num_batches} batches"
            )
            logging.info(
                f"we now need to split the validated folder into {num_batches} subdirectories"
            )

            # Move files in batches
            logging.info(f"Moving files in batches of {batch_size} and aligning them")
            for batch_number in tqdm(range(num_batches)):
                # Create a subdirectory for the current batch
                batch_folder = os.path.join(
                    path_to_validated, f"batch_{batch_number+1}"
                )
                os.makedirs(batch_folder, exist_ok=True)

                # Determine the range of files to move for the current batch
                start_index = batch_number * batch_size
                end_index = start_index + batch_size
                batch_files = files_in_validated[start_index:end_index]

                # Move each file in the current batch
                for i, file in enumerate(batch_files):
                    mp3_file = file + ".mp3"
                    source_path = os.path.join(path_to_validated, mp3_file)
                    batch_path = os.path.join(batch_folder, mp3_file)
                    shutil.copy(
                        source_path, batch_path
                    )  # Copy the file to the batch folder as a safety measure, could be changed to move
                    lab_file = file + ".lab"
                    if not os.path.exists(os.path.join(path_to_validated, lab_file)):
                        logging.warning(
                            f"Lab file {lab_file} does not exist, skipping this file"
                        )
                        continue
                    else:
                        source_path = os.path.join(path_to_validated, lab_file)
                        batch_path = os.path.join(batch_folder, lab_file)
                        shutil.copy(
                            source_path, batch_path
                        )  # Copy the file to the batch folder as a safety measure, could be changed to move

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
                assert (
                    run.returncode == 0
                ), f"The aligner did not run successfully for batch {batch_number +1} that was run"
                logging.info(f"Batch {batch_number +1} was aligned successfully")
                shutil.rmtree(
                    batch_folder
                )  # once the batch is done, remove the folder only makes sense in the case of copying
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
            assert (
                run.returncode == 0
            ), "The aligner did not run successfully for the single batch that was run"
        logging.info("The aligner ran successfully")

    def check_structure(self, word_amount, min_word_count):
        """
        Checks if the corpus has the right  and if not corrects this

        Params:
        min_word_count (int): The minimum amount of words a word should have to be included in  the word amount argument
        word_amount (int): How many words should be processed, if 0 all words are processed, inclusion is based on the min_word_count argument

        Returns:
         clipnames: List[str]
            A list of the clip names
          Sentence_list: List [str]
          A list of  the transcriped sentences in the same order as the clips.
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
        clip_names, sentence_list = self.format_corpus(word_amount, min_word_count)
        missing_clips = set(clip_names).difference(lab_files.intersection(mp3_files))
        assert (
            missing_clips == set()
        ), f"The lab files and mp3 files do not match, since the following clips are missing: {missing_clips}"
        return clip_names, sentence_list

    def create_data_frame_mp(self, clip_list: list, sentence_list: list, num_cores):
        """
        Creates Dataframe with Vocaltract Lab data and other data with multiprocessing
        Parameters:
        clip_list (list): A list of the clip names present in the corpus
        sentence_list (list): A list of the sentences present in the corpus in the same order as the clip_list , so they fit together
        num_cores (int): The number of cores to maximaly use

        Returns:
        pd.dataframe: A dataframe with the following labels
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

        logging.info(
            f"Starting to create the dataframe with multiprocessing using {num_cores} cores"
        )
        with tqdm(total=len(clip_list)) as pbar:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:

                try:
                    futures = list(
                        (
                            executor.submit(
                                generate_rows,
                                clip,
                                sentence,
                                self.path_to_corpus,
                                self.language,
                                self.word_amount,
                                self.word_set,
                            )
                            for clip, sentence in zip(clip_list, sentence_list)
                        )
                    )
                    for future in as_completed(futures):
                        pbar.update(1)
                    results = [f.result() for f in futures]

                except KeyboardInterrupt:
                    logging.warning("Ctrl+C detected, shutting down...")

                    executor.shutdown(wait=True, cancel_futures=True)
                    logging.warning("All processes terminated.")

        logging.info("All processes terminated. Now concatenating the results")

        df = pd.concat(results)

        if os.path.exists(os.path.join(self.path_to_corpus, "clips/temp_output")):
            shutil.rmtree(
                os.path.join(self.path_to_corpus + "/clips/temp_output")
            )  # we don't need the temp_output files anymore

            logging.info("Removed temp_output folder,to clean up space")
        else:
            logging.info("Temp_output folder was not removed, because it was not found")

        return df

    def create_data_frame(
        self,
        clip_list: list,
        sentence_list: list,
    ):
        """
        Creates Dataframe with Vocaltract Lab data and other data
        Parameters:
        path_to_corpus (str): The path to the corpus
        clip_list (list): A list of the clip names present in the corpus
        sentence_list (list): A list of the sentences present in the corpus in the same order as the clip_list , so they fit together

        Returns:
        pd.dataframe: A dataframe with the following labels
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
        
        # remove extension for TextGrid

        path_to_aligned = os.path.join(self.path_to_corpus, "clips_aligned")
        for filename_no_extension, sentence in tqdm(
            zip(clip_list, sentence_list), total=len(clip_list)
        ):

            clip_name = filename_no_extension + ".mp3"

            target_audio, sampling_rate = sf.read(
                os.path.join(self.path_to_corpus, "clips_validated", clip_name)
            )

            assert (
                len(target_audio.shape) == 1
            ), f"The audio file {clip_name} is not mono"
            try:
                tg = textgrid.openTextgrid(
                    os.path.join(path_to_aligned, filename_no_extension + ".TextGrid"),
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

                if  word.label not in self.word_set:
                    logging.info(
                        f"Word '{word.label}' is not in the word set, skipping this word"
                    )
                    continue
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
                    continue
                lexical_word = replace_special_chars(
                    split_sentence[word_index]
                )  # remove special characters  since the basic sematic meaning is not changed by them ( might impact pronunciation however)
                lexical_words.append(lexical_word)
                assert (
                    word.label.lower().replace("'", "") == lexical_word.lower()
                ), f"Word mismatch since word_label: '{word.label.lower().replace("'", "")}' is not equal to lexical_word: '{lexical_word.lower()}' in sentence '{sentence}' in {filename_no_extension}. TextGrid sentece: {text_grid_sentence}"
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
                path = os.path.join(self.path_to_corpus, "clips")
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

                # this is for manual testing only
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

                if (
                    self.path_to_corpus == "../../mini_corpus"
                    or self.path_to_corpus == "../../miniatur_korpus"
                ):
                    if not os.path.exists(
                        f"manual_tests/synthesized_clips_{self.language}"
                    ):
                        os.mkdir(f"manual_tests/synthesized_clips_{self.language}")
                    sf.write(
                        f"manual_tests/synthesized_clips_{self.language}/{word.label}.wav",
                        wav_syn,
                        wav_syn_sr,
                    )

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
        if not os.path.exists(os.path.join(self.path_to_corpus, "clips/temp_output")):
            logging.warning(
                "Temp_output folder was not removed, because it was not found"
            )
        else:
            shutil.rmtree(
                os.path.join(self.path_to_corpus + "/clips/temp_output")
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
        default="../../mini_corpus/",
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
        "--use_mp",
        action="store_true",
        default=False,
        help="if multiprocessing should be used in the creation of the dataframe",
    )
    parser.add_argument(
        "--search_df",
        action="store_true",
        default=False,
        help=" If a already created dataframe should be searched for and then used instead of creating a new one",
    )
    parser.add_argument(
        "--df_path",
        type=str,
        default=None,
        help="The path to the dataframe that should be expaned",
    )  # TODO

    parser.add_argument(
        "--min_word_count",
        type=int,
        default=0,
        help="The minimum amount of words a word should have to be included in  the word amount argument",
    )
    parser.add_argument(
        "--word_amount",
        type=int,
        default=0,
        help="0 the whole corpus shall be processed, a postivie integer if the number is limited. Since processing is sentence based, more words ( with lower word count), will also be synthesized",
    )

    parser.add_argument(
        "--aligner_batch_size",
        type=int,
        default=5000,
        help="How many text files the aligner should process in one batch",
    )

    parser.add_argument(
        "--num_cores",
        type=int,
        default=1,
        help="The number of jobs the multiprocessing should use, uses maximum on default. If the number is 1 or lower, no multiprocessing is used",
    )

    parser.add_argument("--save_df_path", type=str, default="corpus_as_df_mp", help="The path to save the dataframe to in relation to the corpus folder")
    parser.add_argument("--debug", action="store_true", default=False, help="If debug mode should be used")

    parser.add_argument("--epoch_size", type=int, default=10000, help="The size of the epochs used until the dataframe is saved")
    parser.add_argument("--start_epoch", type=int, default=0, help="The epoch to start with (inclusive)")
    parser.add_argument("--end_epoch", type=int, default=None, help="The epoch to end with (inclusive)")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    assert os.path.isdir(args.corpus), "The provided path is not a directory"

    CreateCorpus.setup(language=args.language)
    corpus_worker = CreateCorpus(args.corpus, language=args.language)
    if args.search_df:
        pass
    clip_list, sentence_list = corpus_worker.check_structure(
        args.word_amount, args.min_word_count
    )
    if args.needs_aligner:
        mfa_workers = args.mfa_workers
        corpus_worker.run_aligner(mfa_workers, args.aligner_batch_size)


    clip_lists = [clip_list[i:i + args.epoch_size] for i in range(0, len(clip_list), args.epoch_size)]
    sentence_lists = [sentence_list[i:i + args.epoch_size] for i in range(0, len(sentence_list), args.epoch_size)]
    logging.info(f"Epochs: {len(clip_lists)}")

    folder_path = os.path.join(args.corpus, args.save_df_path +"_folder")
    if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    for i, (clip_list, sentence_list) in enumerate(zip(clip_lists, sentence_lists)):
        if i < args.start_epoch:
            continue
        if args.end_epoch is not None:
            if i > args.end_epoch:
                break
        if args.use_mp:
                if args.num_cores <= 1:
                    assert args.num_cores >= 0, "The number of cores cannot be negative"
                    logging.info(
                        f" You want to use multiprocessing but the number of cores is {args.num_cores}, so the mulitprocessing function likely has no benefit"
                    )
                    logging.info(f"Melspecs will not be created in multiprocessing mode")
                df = corpus_worker.create_data_frame_mp(
                    clip_list, sentence_list, args.num_cores
                )

        else:
            logging.info("Creating dataframe without multiprocessing")
            df = corpus_worker.create_data_frame(clip_list, sentence_list)
        logging.info(df)
       
        path_to_save_corpus = os.path.join(folder_path, args.save_df_path + f"epoch_{i}" + ".pkl") 
        df.to_pickle(path_to_save_corpus)
        logging.info(f"Dataframe saved to {path_to_save_corpus}")
        logging.info(f"Epoch {i} done")


        
    logging.info("Merging all DataFrames into one")
    df_list = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(folder_path):
        # Only process .pkl files
        if filename.endswith(".pkl"):
            file_path = os.path.join(folder_path, filename)
            # Load the .pkl file into a DataFrame and append it to the list
            df = pd.read_pickle(file_path)
            df_list.append(df)
    
    # Concatenate all DataFrames
    if df_list:
        concatenated_df = pd.concat(df_list, ignore_index=True)
        concatenated_df.to_pickle(os.path.join(args.corpus, args.save_df_path + ".pkl"))
    else:
        logging.error("No .pkl files found.")
        
    logging.info(concatenated_df)
    logging.info("Done! :P")
