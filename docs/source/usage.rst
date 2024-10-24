===========
How to use?
===========

``create_vtl_corpus`` allows to automatically create control parameter
trajectories for the VocalTractLab (VTL) artculatory synthesizer for a large
corpus of speech audio files and transcriptions of these files. It includes the
code to align the transcriptions to the audio with the Monreal Forced Aligner
(MFA).  ``create_vtl_corpus`` is mainly used and tested with data from the
Mozilla Common Voice project.

To align and synthesize the first 100 words, which apear at least 4
times, of a German speech corpus at the path ``CORPUS`` and save the results as
a pandas DataFrame to ``SAVE_DF_PATH`` run the following command:

.. code:: bash

    python -m create_vtl_corpus.create_corpus --corpus CORPUS --language de --needs_aligner --use_mp --min_word_count 4 --word_amount 100 --save_df_name SAVE_DF_NAME

Use ``--help`` or ``-h`` to get a full list of the command line options.

Furthermore, you can use it within Python. Here the
``create_vtl_corpus.create_corpus.CreateCorpus`` class is a good starting
point.


Use cases
=========
``create_vtl_corpus`` can be used to:

1. Create the vtl corpus with no pre-existing data but not all of the files, i.
   e. only 10_000 samples.

2. Add new data to an already existing corpus, add 10_000 new samples.

3. Filter after words? Create 1000 more "post" word types.



Flags
=====
The following flags can be used to modify the behaviour of the library.

.. argparse::
   :module: create_vtl_corpus.create_corpus
   :func: return_argument_parser
   :prog: fancytool

   
Multiprocessing
===============
The library supports multiprocessing, which can be used to speed up the process
and for large corpora this is absolutely necessary.  However it is not enabled
by default, to enable it use the ``--use_mp`` flag.  For small corpora it is
not recommended to use multiprocessing, since generating the melspectrograms is
not possible with multiprocessing currently.  The melspectrograms can be
generated afterwards too however with the information available in the
dataframe, but no solution is provided by the library for this yet.


CreateCorpus Class
==========================

.. autoclass:: create_vtl_corpus.create_corpus.CreateCorpus
   :members:
   



