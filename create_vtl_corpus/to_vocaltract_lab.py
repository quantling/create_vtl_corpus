"""
This file generates the vocaltract lab trajectories for a corpus. 
It's assumed that a corpus has the following shape 
 as is common with Mozillas Common Voice Corpus)
 This file uses the Montreal Forced Aligner 

corpus/
├── validated.txt          a file where the transripts are stored
|
├── clips/
|     └──*.mp3
└── *.files_not_relevant_to this project


"""

import pandas as pd
import os
import argparse
import subprocess


class CreateVocaltractLab:

    def __init__(self, path_to_corpus: str, language: str):
        self.path_to_corpus = path_to_corpus
        self.language = language

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

        for _, row in data.iterrows():
            transcript = row["sentence"]
            file_name = row["path"].removesuffix(".mp3") + ".lab"
            with open(
                os.path.join(self.path_to_corpus, "clips", file_name), "wt"
            ) as lab_file:
                lab_file.write(transcript)

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
                "english_us_arpa",
                "english_us_arpa",
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

        subprocess.run(command)

    def check_structure(self):
        """
        Checks if the corpus has the right  and if not corrects this

        Params:
        -

        Returns:
         -
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
            return
        else:
            print("The lab files and mp3 files do not match, correcting this now")
            self.format_corpus()


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
    args = parser.parse_args()

    assert os.path.isdir(args.corpus), "The provided path is not a directory"

    vtl = CreateVocaltractLab(args.corpus, args.language)
    vtl.check_structure()
    vtl.run_aligner()
