========
Glossary
========
Collection of terms and their definition within the synthesis pipelines and the
creation of the corpus for a vocal tract synthesis with VocalTractLab.


VocalTractLab (VTL)
===================
VocalTractLaboratory (VTL) is the articulatory speech syntesizer developed by Peter Birkholz.
You can find more informaton about VTL on the `VTL website <https://www.vocaltractlab.de/>`_.


Control Parameter Trajectories (cp-trajectories)
================================================


Montreal Forced Aligner (MFA)
=============================
The Montreal Forced Aligner (MFA) is a speech processing tool that aligns speech to its transcript.
It is developed by MontrealCorpusTools at McGill university in Montreal. You can find more information about MFA on the `MFA website <https://montreal-forced-aligner.readthedocs.io/en/latest/>`_.  


Phonetic alphabet
=================


 You can see the phonetic alphabet used by VTL here :ref:`Phonemes` .
SAMPA
=====
SAMPA (Speech Assessment Methods Phonetic Alphabet) is a computer-readable phonetic script using 7-bit ASCII characters.
It is used to represent the phonemes of a language. You can find more information about SAMPA on the `SAMPA website <http://www.phon.ucl.ac.uk/home/sampa/home.htm>`_.
It is used by VTL to synthesize.

MFA alphabet
============
The phonetic alphabet used by default by the MFA is a opinionated version of
IPA symbols. With the create_vtl_corpus we target this alphabet in the
segment-based synthesis with VTL. You can read more about this alphabet in the
`MFA documentation <https://mfa-models.readthedocs.io/en/latest/mfa_phone_set.html>`_.


Segment-based synthesis
=======================


PAULE synthesis
===============

Predictive Articulatory speech synthesis Utilizing Lexical Embeddings (PAULE) is a python frame work to plan control parameter trajectories 
for the VocalTractLab simulator for a target acoustics or semantic embedding developed by Tino Sering . 


Melspectrograms
===============
A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale. The mel scale is a perceptual scale of pitches that is based on the human ear's response to different frequencies.
We call this function imported from PAULE to create melspectrograms from the wav files and the synthesized audio.

We first resample then create the melspectrogram and then create the melspectrogram in decibels

.. code :: python

    def librosa_melspec(wav, sample_rate):
    wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=44100,
            res_type='kaiser_best', fix=True, scale=False)
    melspec = librosa.feature.melspectrogram(y=wav, n_fft=1024, hop_length=220, n_mels=60, sr=44100, power=1.0, fmin=10, fmax=12000)
    melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)
    return np.array(melspec_db.T, order='C', dtype=np.float64)


