========
Pipeline
========

Some conceptual thoughts on the processing pipeline to generate the control
parameter trajectories, splice out the single words and do the segment-based
synthesis.

Goal
======
The goal of the pipeline is to generate a corpus for a vocal tract synthesis with VocalTractLab (VTL) from the Mozilla Common Voice corpus. The corpus is specifically tailored
for the `PAULE model <https://github.com/quantling/paule>`__  but it can be used for other purposes.


Output Format
=============
As the final output create_vtl_corpus generates a pandas.DataFrame with the following columns:

* 'file_name' : name of the mp3 file in the common voice corpus
* 'word_type' : word type, i. e. type of the word in terms of graphemic transcription
* 'word_position' : postion of the word type in the sentence
* 'sentence' : transcription of the full sentence
* 'wav_recording' : spliced out audio as mono audio signal
* 'sr_recording' : sampling rate of the recording
* 'sampa_phones' : list of phones in sampa notation
* 'phone_durations' : list of durations of the phones
* 'vector' : fastText vector embedding for the word_type
* 'cp_segment' : cp-trajectories of the segment-based synthesis

The following columns are added, even if they can be generated out of the entries we already have for convenience:

* 'wav_segment' : wave form as mono audio from the segment-based synthesis
* 'sr_segment' : sampling rate for the mono audio from the segment-based synthesis
* 'melspec_recording' : acoustic representation of human recording of the common voice corpus (log-mel spectrogram)
* 'melspec_segment' : acoustic representation of the segment-based approach (log-mel spectrogram)


Pipeline
========
The idea of the processing pipeline is:

1. align the audio corpus and transcriptions with the MFA
2. extract the word types and splice out the audio
3. extract the phones and phone durations from the alignment
#. convert stereo audio to mono
#. extract the pitch of the audio signal
#. generate gestural scores with the segment-based approach in VTL
#. fit the pitch with the targetoptimizer
#. merge the f0 gesture of the targetoptimizer to the gestural scores of the
   segment-based approach
#. synthesize cp-trajectories from the patched gestural scores
#. synthesize audio (wav_segment, sr_segment) from the cp-trajectories
#. retrieve the fastText embedding vector for the word type
#. calculate the aucoustic representation (log-mel-spectrogram) for the wav_recording, and wav_segment


Notes
=====
Some random notes to keep in mind.

* pauses between words are dropped
* we use the MFA (IPA like) phonemes and not the ARPA ones and then convert them to the SAMPA like phonemes needed for VTL


Resources
=========
The following resources are used:

*  `VocalTractLab <https://vocaltractlab.de/>`__ (use the version included in create_vtl_corpus)
* targetoptimizer (use the version included in create_vtl_corpus)
* `Montreal forced aligner  <https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html>`__
* `Mozilla Common Voice <https://commonvoice.mozilla.org/en>`__
* `fastText word embedding model <https://fasttext.cc/>`__



Phonemes 
========
The phonemes are converted from the MFA phonemes to the SAMPA phonemes. The following table shows the conversion:

.. list-table:: Phoneme to VTL Representation
   :widths: 15 15
   :header-rows: 1

   * - Phoneme
     - VTL Representation
   * - a
     - a
   * - aj
     - aI
   * - aw
     - aU
   * - aː
     - a:
   * - b
     - b
   * - bʲ
     - b
   * - c
     - k
   * - cʰ
     - s
   * - d
     - d
   * - dʒ
     - dZ
   * - dʲ
     - d
   * - e
     - e
   * - ej
     - I
   * - f
     - f
   * - fʲ
     - f
   * - h
     - h
   * - i
     - i
   * - iː
     - i:
   * - j
     - j
   * - k
     - k
   * - kʰ
     - k
   * - l
     - l
   * - m
     - m
   * - mʲ
     - m
   * - m̩
     - m
   * - n
     - n
   * - n̩
     - n
   * - o
     - o
   * - ow
     - aU
   * - p
     - p
   * - pʰ
     - p
   * - pʲ
     - p
   * - s
     - s
   * - t
     - t
   * - tʃ
     - tS
   * - tʰ
     - t
   * - tʲ
     - t
   * - u
     - u
   * - uː
     - u:
   * - v
     - v
   * - vʲ
     - v
   * - w
     - U
   * - z
     - z
   * - æ
     - a
   * - ç
     - C
   * - ð
     - D
   * - ŋ
     - N
   * - ɐ
     - 6
   * - ɑ
     - o
   * - ɑː
     - o:
   * - ɒ
     - O
   * - ɒː
     - O
   * - ɔ
     - O
   * - ɔj
     - OY
   * - ə
     - @
   * - əw
     - aU
   * - ɚ
     - @
   * - ɛ
     - E
   * - ɛː
     - E:
   * - ɜ
     - 2
   * - ɜː
     - 2:
   * - ɝ
     - 2
   * - ɟ
     - dZ
   * - ɡ
     - g
   * - ɪ
     - I
   * - ɫ
     - l
   * - ɫ̩
     - l
   * - ɱ
     - m
   * - ɲ
     - n
   * - ɹ
     - r
   * - ɾ
     - r
   * - ʃ
     - S
   * - ʉ
     - u
   * - ʉː
     - u:
   * - ʊ
     - U
   * - ʎ
     - l
   * - ʒ
     - Z
   * - ʔ
     - ?
   * - θ
     - T
   * - ʁ
     - R
   * - eː
     - e:
   * - x
     - x
   * - ts
     - ts
   * - ɔʏ
     - OY
   * - oː
     - o:
   * - œ
     - 9
   * - yː
     - y:
   * - ʏ
     - Y
   * - øː
     - 2:
   * - ø
     - 2
   * - pf
     - pf
   * - l̩
     - l
   * - t̪
     - T
   * - ʈʲ
     - T
   * - ʈ
     - t
   * - ʋ
     - v
   * - d̪
     - d
   * - kʷ
     - k
   * - cʷ
     - C
   * - ɖ
     - d
   * - tʷ
     - t
   * - ɟʷ
     - dZ



Some phonemes are perhaps not perfectly converted, since VTL does not accept all the phonemes of the SAMPA notation. Also, the MFA phonemes are not always perfectly aligned with the SAMPA phonemes.
If VTL accepts more phonemes in the future, the conversion can be improved. Please contact the author if you have suggestions.
The conversion should be good enough for the purpose of the corpus generation.
A german accent in English is  noticable in English pronounciation in the synthesis.
If other languages are added the conversion table must be adapted for new phonemes.
