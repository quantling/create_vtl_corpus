======
Readme
======

.. image:: https://zenodo.org/badge/167427297.svg
   :target: https://zenodo.org/badge/latestdoi/167427297

This package supplies the necessary functions in order to synthesize speech
from a phonemic transcription. Furthermore, it defines helpers to improve the
result if more information as the pitch contour is available. It is especially useful when working with 
the `PAULE <https://github.com/quantling/paule>`__ framework.

Currently the package supports the following languages:
   - German
   - English

Version 2.0.0 and later
-----------------------
From version 2.0.0 we are relying on the new segment-to-gesture API introduced
in VTL 2.3 and use the JD3.speaker instead of the JD2.speaker.

Old version 1.1.0
-----------------
The original version of this tool is based on the work and on the Matlab code
on Yingming Gao. This can be viewed by checking out the tag ``1.1.0``.

The overall logic is in ``create_corpus.py`` which executes the appropriate
functions from top to bottom. The functions are supplied by the other files.

.. note::

   In the since VTL version 2.3 which can be downloaded as free software from
   https://www.vocaltractlab.de/index.php?page=vocaltractlab-download most of
   the functionality implemented here is available directly from the VTL api.
   Please use the VTL api directly.


Minimal Example
===============
Given a german Corpus with the following structure which is what the `Mozilla Common Voice project <https://commonvoice.mozilla.org>`__ provides:

 .. code:: bash

    corpus/
    ├── validated.tsv         # a file where the transcripts are stored
    ├── clips/
    │   └── *.mp3             # audio files (mp3)
    └── files_not_relevant_to_this_project

If you run the following command the package will align the audio files for you, and then create a pandas DataFrame with the synthesized audio and other information useful for the PAULE model,
but only for the first 100 words that occur 4 times or more. Since you use multiprocessing, no melspectrograms are generated.:

.. code:: bash

    python -m create_vtl_corpus.create_corpus --corpus CORPUS --language de --needs_aligner --use_mp --min_word_count 4 --word_amount 100 --save_df_name SAVE_DF_NAME

The end product should look someting like this

.. code:: bash

   corpus/
   ├── validated.tsv          # a file where the transcripts are stored
   ├── clips/
   │   ├── *.mp3              # mp3 files
   │   └── *.lab              # lab files
   ├── clips_validated/
   │   ├── *.mp3              # validated mp3 files
   │   └── *.lab              # validated lab files
   ├── clips_aligned/
   │   └── *.TextGrid         # aligned TextGrid files
   ├── corpus_as_df.pkl       # a pandas DataFrame with the information
   └── files_not_relevant_to_this_project

The DataFrame contains the following columns

.. list-table:: Dataframe Labels
   :header-rows: 1

   * - Column Name
     - Description
   * - file_name
     - Name of the clip
   * - label
     - The spoken word as it is in the aligned textgrid
   * - lexical_word
     - The word as it is in the dictionary
   * - word_position
     - The position of the word in the sentence
   * - sentence
     - The sentence the word is part of
   * - wav_recording
     - Spliced out audio as mono audio signal
   * - sr_recording
     - Sampling rate of the recording
   * - sr_synthesized
     - Sampling rates synthesized
   * - sampa_phones
     - The SAMPA(like) phonemes of the word
   * - mfa_phones
     - The phonemes as outputted by the aligner
   * - phone_durations_lists
     - The duration of each phone in the word as list
   * - cp_norm
     - Normalized CP-trajectories
   * - vector
     - Embedding vector of the word, based on FastText Embeddings
   * - client_id
     - ID of the client


Copyright
=========
As the VocalTractLabAPI.so and the JD2.speaker is under GPL v3 the rest of the code
here is GPL  under as well.  If the code is not dependent on VTL anymore you can use
it under MIT license.


Citing 
=======
If you use this code for your research, please cite the following thesis:

Konstantin Sering. Predictive articulatory speech synthesis utilizing lexical embeddings (PAULE). PhD thesis, Universität Tübingen, 2023.

.. code:: bibtex
   
      @phdthesis{sering2023paule,
         title={Predictive articulatory speech synthesis utilizing lexical embeddings (PAULE)},
         author={Sering, Konstantin},
         year={2023},
         school={Universität Tübingen}
      }

   

Acknowledgments
===============
This research was supported by an ERC advanced Grant (no. 742545), by the
University of Tübingen and by the TU Dresden.

