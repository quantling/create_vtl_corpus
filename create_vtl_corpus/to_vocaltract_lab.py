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


CORPUS = "../../mini_corpus"  # this is the path to our corpus


LANGUAGE = "en"  # en or german are possible


def format_corpus(path_to_corpus: str):
    """
    Takes the path to the corpus and formats it to the fitting form

    Params:
    path_to_corpus (str): The path to the corpus

    Returns:
    -
    """

    data = pd.read_table(os.path.join(path_to_corpus, "validated.tsv"), sep="\t")

    for _, row in data.iterrows():
        transcript = row["sentence"]
        file_name = row["path"].removesuffix(".mp3") + ".lab"
        with open(os.path.join(path_to_corpus, "clips", file_name), "wt") as lab_file:
            lab_file.write(transcript)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a corpus to the vocaltract lab format"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=CORPUS,
        help="The path to the corpus which should be converted to the vocaltract lab format",
    )
    args = parser.parse_args()
    format_corpus(args.corpus)
    print("Done")
