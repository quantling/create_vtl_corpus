import pandas as pd
import os
import argparse
import subprocess
import ctypes
import fasttext
import contextlib

from paule import util
from praatio import textgrid
import soundfile as sf


class CreateVocalTractLab:
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
    extract_sampa():
        Extracts the SAMPA phonemes from the aligned corpus
    """

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
            "bʲ": "b'",
            "c": "c",
            "cʰ": "c",  # c_h not possible in SAMPA but in X SAMPA
            "d": "d",
            "dʒ": "dZ",
            "dʲ": "d'",
            "e": "e",
            "ej": "eI",
            "f": "f",
            "fʲ": "f'",
            "h": "h",
            "i": "i",
            "iː": "i:",
            "j": "j",
            "k": "k",
            "kʰ": "k",  # "k_h", is not possible in SAMPA but in X SAMPA
            "l": "l",
            "m": "m",
            "mʲ": "m'",
            "m̩": "m%",
            "n": "n",
            "n̩": "n%",
            "o": "o",
            "ow": "oU",
            "p": "p",
            "pʰ": "p",  # "p_h", is not possible in SAMPA ibut in X SAMPA
            "pʲ": "p'",
            "s": "s",
            "t": "t",
            "tʃ": "tS",
            "tʰ": "t",  # "t_h", is not possible in SAMPA but in X SAMPA
            "tʲ": "t'",
            "u": "u",
            "uː": "u:",
            "v": "v",
            "vʲ": "v'",
            "w": "w",
            "z": "z",
            "æ": "{",
            "ç": "C",
            "ð": "D",
            "ŋ": "N",
            "ɐ": "6",
            "ɑ": "A",
            "ɑː": "A:",
            "ɒ": "Q",
            "ɒː": "Q:",
            "ɔ": "O",
            "ɔj": "OI",
            "ə": "@",
            "əw": "@U",
            "ɚ": "@`",
            "ɛ": "E",
            "ɛː": "E:",
            "ɜ": "3",
            "ɜː": "3:",
            "ɝ": "3`",
            "ɟ": "J",
            "ɡ": "g",
            "ɪ": "I",
            "ɫ": "5",
            "ɫ̩": "5=",
            "ɱ": "F",
            "ɲ": "J",
            "ɹ": "r",
            "ɾ": "4",
            "ʃ": "S",
            "ʉ": "}",
            "ʉː": "}:",
            "ʊ": "U",
            "ʎ": "L",
            "ʒ": "Z",
            "ʔ": "?",
            "θ": "T",
        }

    def load_fasttext_model(self, language: str):
        """
        Loads the fasttext model for the given language

        Params:
        language (str): The language of the model

        Returns:
        fasttext.FastText._FastText: The loaded fasttext model
        """
        if language == "en":
            model = fasttext.load_model("../../cc.en.300.bin")
        elif language == "de":
            model = fasttext.load_model("../../cc.de.300.bin")
        else:
            raise ValueError("The language is not supported")
        print("Fasttext model loaded")
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
        for _, row in data.iterrows():
            transcript = row["sentence"]
            clip_name = row["path"].removesuffix(".mp3")
            clip_names.append(clip_name)
            file_name = clip_name + ".lab"
            with open(
                os.path.join(self.path_to_corpus, "clips", file_name), "wt"
            ) as lab_file:
                lab_file.write(transcript)
        return clip_names

    def run_aligner(self):
        """
         Runs the Montreal Forced Aligner on the corpus

        Params:
        -

        Returns:
        -
        """
        if self.language == "en":
            print("aligning corpus in english")
            command = "conda run -n aligner mfa  align".split() + [
                os.path.join(self.path_to_corpus, "clips"),
                "english_mfa",
                "english_mfa",
                os.path.join(self.path_to_corpus + "_aligned"),
            ]

        if self.language == "de":
            print("aligning corpus in german")
            command = "conda run -n aligner mfa  align".split() + [
                os.path.join(self.path_to_corpus, "clips"),
                "german_mfa",
                "german_mfa",
                os.path.join(self.path_to_corpus + "_aligned"),
            ]

        run = subprocess.run(command)
        assert run.returncode == 0, "The aligner did not run successfully"

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

        if lab_files == mp3_files:

            clip_names = lab_files
            print("The lab files and mp3 files match")

            return sorted(clip_names)
        else:
            print("The lab files and mp3 files do not match, correcting this now")
            clip_names = self.format_corpus()

        return sorted(clip_names)

    def extract_sampas_and_cut_audio(self, path_to_corpus: str, clip_list: list):
        """
        Extracts the sampas and cuts the audio word by word for the whole corpus
        Parameters:
        path_to_corpus (str): The path to the corpus
        clip_list (list): A list of the clip names present in the corpus

        Returns:
        Dataframe: A dataframe with the following labels
        'file_name' : name of the clip
        'label' : the spoken word
        'word_position' : the position of the word in the sentence
        'sentence' : the sentence the word is part of
        'wav_recording' : spliced out audio as mono audio signal
        'sr_recording' : sampling rate of the recording
        'sampa_phones' : the sampa(like) phonemes of the word
        'phone_durations' : the duration of each phone in the word
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
        sampling_rates = list()
        phone_durations = list()
        sampa_phones = list()
        cp_norms = list()
        melspec_norm_recordeds = list()
        melspec_norm_synthesizeds = list()
        vectors = list()
        client_ids = list()

        # remove extension for TextGrid
        for filename_no_extension in clip_list:
            clip_name = filename_no_extension + ".mp3"
            print(clip_name)
            target_audio, sampling_rate = sf.read(
                os.path.join(path_to_corpus, "clips", clip_name)
            )

            assert (
                len(target_audio.shape) == 1
            ), f"The audio file {clip_name} is not mono"
            tg = textgrid.openTextgrid(
                os.path.join(
                    path_to_corpus + "_aligned", filename_no_extension + ".TextGrid"
                ),
                False,
            )
            sentence_list = []
            for word_index, word in enumerate(tg.getTier("words")):
                sentence_list.append(word.label)
            sentence = " ".join(sentence_list)
            for word_index, word in enumerate(tg.getTier("words")):

                phones = list()

                # adding easy to add variables to the lists
                labels.append(word.label)
                sampling_rates.append(sampling_rate)
                word_positions.append(word_index)
                fasttext_vector = self.fast_text_model.get_word_vector(word.label)
                vectors.append(fasttext_vector)
                client_ids.append(filename_no_extension)
                sentences.append(sentence)

                phone_durations = list()
                for phone in tg.getTier("phones").entries:
                    if phone.label == "spn":
                        break
                    if phone.start >= word.end:
                        break
                    if phone.start < word.start:

                        continue

                    print(word.label)
                    print("MFA Phones", phone.label)
                    phones.append(self.mfa_to_sampa_dict[phone.label])
                    print("sampa phones", phones)
                    phone_durations.append(phone.end - phone.start)

                if not phones:
                    continue

                # splicing audio
                wav_rec = target_audio[
                    int(word.start * sampling_rate) : int(word.end * sampling_rate)
                ]
                wavs.append(wav_rec)
                # write seg file
                rows = []
                for i, phone in enumerate(phones):
                    row = "name = %s; duration_s = %f;" % (phone, phone_durations[i])
                    rows.append(row)
                text = "\n".join(rows)
                path = os.path.join(path_to_corpus + "_aligned", "clips")
                if not os.path.exists(path):
                    os.mkdir(path=path)
                # delete this later
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
                melspec_norm_recordeds.append(melspec_norm_rec)
                wav_syn, wav_syn_sr = util.speak(cps)
                melspec_norm_syn = util.normalize_mel_librosa(
                    util.librosa_melspec(wav_syn, wav_syn_sr)
                )

                melspec_norm_syn = util.pad_same_to_even_seq_length(melspec_norm_syn)
                melspec_norm_synthesizeds.append(melspec_norm_syn)

        df = pd.DataFrame(
            {
                "file_name": clip_list,
                "label": labels,
                "word_position": word_positions,
                "sentence": sentences,
                "wav_recording": wavs,
                "sr_recording": sampling_rates,
                "sampa_phones": sampa_phones,
                "phone_durations": phone_durations,
                "cp_norm": cp_norms,
                "melspec_norm_recorded": melspec_norm_recordeds,
                "melspec_norm_synthesized": melspec_norm_synthesizeds,
                "vector": vectors,
                "client_id": client_ids,
            }
        )
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a corpus to the vocaltract lab format"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="../../mini_corpus",
        help="The path to the corpus which should be converted to the vocaltract lab format",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="The language of the corpus as an abbreviation",
    )
    parser.add_argument("--needs_aligner", action="store_true", default=False)
    args = parser.parse_args()

    assert os.path.isdir(args.corpus), "The provided path is not a directory"

    vtl = CreateVocaltractLab(args.corpus, language=args.language)
    clip_list = vtl.check_structure()
    if args.needs_aligner:
        vtl.run_aligner()
    print(clip_list)
    vtl.extract_sampas_and_cut_audio(args.corpus, clip_list)
    print("Done! :P")
