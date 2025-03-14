import pandas as pd
import os

import pickle
import time
from paule import util
import argparse
import psutil

def add_mels_to_df(files, skip_index, data_path, make_slim):
    for i,file in enumerate(files):
        if i < skip_index:
            print(f"Skipping {file}")
            continue
        print(f" use skip index {i} to skip to here")
        corpus_path = os.path.join(data_path,file)
        print(f"loading {corpus_path}")
        with open(
            corpus_path,
            "rb",
        ) as pickle_file:
            df = pickle.load(pickle_file)

        start = time.time()
        print(f"Adding  spectrograms to {file}")
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
        if make_slim:
            print("Making slim version of the dataframe")
            df = df[[ "wav_synthesized", "melspec_norm_synthesized", "melspec_norm_recorded", "lexical_word", "vector", "cp_norm"]]
        temp_path = corpus_path + "_temp"
        print(f"Saving the dataframe to {corpus_path}")
        df.to_pickle(temp_path)
        os.rename(temp_path, corpus_path)
        print(f"Saved the dataframe to {corpus_path} and removed the temp file")
        del df
        print(f"completed adding mel spectrograms to {file}")
        print(f"Memory usage: {psutil.virtual_memory().percent}%")
        print(f"Memory usage: {psutil.virtual_memory().used/1024**3}GB")
        print(f" used disk space: {psutil.disk_usage(data_path).percent}%")
        print(f" used disk space: {psutil.disk_usage(data_path).used/1024**3}GB")
   
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add mel spectrograms to a dataframe")
    parser.add_argument(
        "files",
        type=str,
        
        help="The path to the pickle files containing the dataframe",
        choices=["corpus_as_df", "full_split", "train", "val", "test", "val+train"],
    )
    parser.add_argument("--skip_index", type = int, default = 0, help = "Skip the first n files")
    parser.add_argument("--language", type = str, default = "de", help = "The language of the dataset")
    parser.add_argument("--data_path", type = str, default = "../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_", help = "The path to the data folder")
    parser.add_argument("--make_slim", action="store_true", help="Make a slim version of the dataframe")
    
    data_path = parser.parse_args().data_path + parser.parse_args().language 
    args = parser.parse_args()
    files = []
    if args.files == "corpus_as_df":
        sorted_files = sorted(os.listdir(data_path))
        filtered_files = [
        file
        for file in sorted_files
        if file.startswith("corpus_as_df_mp") and file.endswith(".pkl")
    ]
        file_names = filtered_files
    if args.files == "full_split":
        files = os.listdir(data_path)
        filtered_files = sorted([ file for file in files if (file.endswith(".pkl") and  "_data_"  in file  )])
        test_files = [file for file in filtered_files if "test" in file]
        validation_files = [file for file in filtered_files if  "validation" in file]
        training_files = [file for file in filtered_files if   "training" in    file]
        file_names = training_files + validation_files + test_files
    if args.files == "train":
        files = os.listdir(data_path)
        filtered_files = sorted([ file for file in files if (file.endswith(".pkl") and  "_data_"  in file  )])
        training_files = [file for file in filtered_files if   "training" in    file]
        file_names = training_files
    if args.files == "val":
        files = os.listdir(data_path)
        filtered_files = sorted([ file for file in files if (file.endswith(".pkl") and  "_data_"  in file  )])
        validation_files = [file for file in filtered_files if  "validation" in file]
        file_names = validation_files
    if args.files == "test":
        files = os.listdir(data_path)
        filtered_files = sorted([ file for file in files if (file.endswith(".pkl") and  "_data_"  in file  )])
        test_files = [file for file in filtered_files if "test" in file]
        file_names = test_files
    if args.files == "val+train":
        files = os.listdir(data_path)
        filtered_files = sorted([ file for file in files if (file.endswith(".pkl") and  "_data_"  in file  )])
        training_files = [file for file in filtered_files if "train" in file]
        validation_files = [file for file in filtered_files if  "validation" in file]
        file_names = validation_files
        file_names = file_names + training_files

    

    print(file_names)
    if args.make_slim:
        print("Making slim version of the dataframes")
    add_mels_to_df(file_names, args.skip_index, data_path, args.make_slim)
    print("All mel spectrograms added")