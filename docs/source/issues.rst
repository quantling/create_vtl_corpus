=====================================
Troubleshooting and known issues
=====================================

Problems with the aligner
==========================
Sometimes the aligner will not generate phonemes for the whole sentence in our setup.
Since we rely on these phonemes to synthesise the audio, we have to return a empty dataframe for the sentence this occurs in with a warning.
On other occasions the aligner will not generate a textgrid file for the sentence. In this case an empty dataframe is also returned with a warning.
Those occasions are not common however.


Seperating words 
================
We rely on the aligner for word-seperation. It's rules are not concistent with written language. Since we want to preserve capitalization and want to return a lexical labe for each word,
we split the lab files by spaces, punctuation and other characters that are not letters. Since the aligner does not seem to follow any concistent rules for both languages currently  implemented this
can lead to misalignment of lexical labels and the labels returned by the aligner.  If this missaligment occurs we return an empty dataframe for this sentence so we can be sure that lexical labels and labels align, when returned in the dataframe.
currently, at least for multiprocessing we also do not check if the words in the transcript are the same as the words in the transcript given by aligner, this is a possible future improvement.
While they align in the majority of cases expect some fringe cases where they do not align.
Furthermore,when words in the whole corpus are counted for words and then put in a frozen set. 

Estimations of lost words and amount of total words in the corpus
=================================================================
Words that would be skipped because they occur less than 4 times in the corpus are not counted in the total amount of words in the corpus or in the amount lost words.
The amount of words in the corpus and the amount of lost words are only estimated if a word seperation error occurs. We then simply split the transcript by spaces and count the words and divide them by a factor of 1.001 since
some of the words in a sentence might not occur 4 times or more, so we would have skipped them anyways. Please adjust the factor for your own corpus if you have a different word distribution and/or a 
minimum word count that is different from our standard 4 times or more.
For other secenarios the words are counted and  we follow the word seperation done by the aligner. 

Estimation of total word types
==============================
We estimate the total word counts based on the lexical words.
Keep this in mind when using the library.

Further issues
==============
If you encounter any other issues, please open an issue on the github page of the project  `here <https://github.com/quantling/create_vtl_corpus/issues/new/choose>`__ . 
Please provide a detailed description of the issue and if possible a minimal example to reproduce the issue.
"""