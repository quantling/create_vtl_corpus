import pandas as pd
import os

import pickle
import time
from paule import util
import argparse


def add_mels_to_df(files, skip_index):
    for i,file in enumerate(files):
        if i < skip_index:
            print(f"Skipping {file}")
            continue

        corpus_path = file

        with open(
            corpus_path,
            "rb",
        ) as pickle_file:
            df = pickle.load(pickle_file)

        start = time.time()
        print("Adding mel spectrograms")
        df["melspec_norm_recorded"] = df["melspec_norm_recorded"] = df.apply(
            lambda row: util.normalize_mel_librosa(
                util.librosa_melspec(row["wav_recording"], row["sr_recording"])
            ),
            axis=1,
        )
        print("Recorded mel spectrograms added")    
        df["melspec_norm_synthesized"] = df.apply(
            lambda row: util.normalize_mel_librosa(
                util.librosa_melspec(row["wav_synthesized"], row["sr_synthesized"])
            ),
            axis=1,
        )
        print("Synthesized mel spectrograms added")
        print(
            f"Mel spectrograms added in {time.time()-start} seconds. This does not use multiprocessing"
        )
        temp_path = corpus_path + "_temp"
        print(f"Saving the dataframe to {corpus_path}")
        df.to_pickle(temp_path)
        os.rename(temp_path, corpus_path)
        del df
        print(f"completed adding mel spectrograms to {file}")
   
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add mel spectrograms to a dataframe")
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="The path to the pickle files containing the dataframe",
        options=["corpus_as_df", "full_split", "train", "val", "test"],
    )
    parser.add_argument("--skip_index", type = int, default = 0, help = "Skip the first n files")

    
        
    args = parser.parse_args()
    files = []
    

    
    add_mels_to_df(files, args.skip_index)