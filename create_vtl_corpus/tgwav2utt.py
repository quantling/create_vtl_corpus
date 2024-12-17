from math import floor
import os

from scipy.io import wavfile
from praatio import tgio

from .textgrid import TextGrid

def create_utterance_split_wave(tg_name, wav_name, utterance_name, wav_dir, text_grid_dir, ii=0):
    """
    Takes a text grid and a wave recording and generates an utterance file and a lot of small wave files, text grids.

    Parameters
    ----------
    tg_name : path to input text grid
    wav_name : path to input wav recording
    utterance_name : path to output utterance file
    wav_dir : path to output wav dir for splitted waves
    text_grid_dir : path to output text grid dir for individual text grids

    Returns
    -------
    ii : int
        number of processed utterances

    Notes
    -----
    - do not take words shorter than 0.150 seconds
    - do not take words with only one or less syllables transcribed

    """

    # read in data
    with open(tg_name, 'rt', encoding='UTF-16') as tg_file:
        fid = TextGrid(tg_file.read())

    phones, syls, words = fid.tiers
    samplerate, wav_data = wavfile.read(wav_name)

    if ii == 0:
        mode = 'wt'
    else:
        mode = 'at'  # append to existing file

    with open(utterance_name, mode) as utt_file:

        if ii == 0:
            # write header
            utt_file.write('Label\tSampa transcription\tPhone durations\n')

        jj = kk = ll = 0

        for wordstart, wordend, word in words.simple_transcript:
         
            #if word in ('<P>', '<LAUGH>', '<UNINTELLIGIBLE>', '<THROATCLEARING>', '<BREATH>', '<COUGH>', '<YAWN>'):
            if '<' in word:
                continue

            wordstart = float(wordstart)
            wordend = float(wordend)

            # generate utterances
            word_syls = []
            durations = []
            while kk < len(syls.simple_transcript) and float(syls.simple_transcript[kk][1]) <= wordend:
                sylsstart, sylsend, _ = syls.simple_transcript[kk]
                sylsstart = float(sylsstart)
                sylsend = float(sylsend)
                kk += 1
                if sylsstart < wordstart:
                    continue
               
                syls_phones = []
                while ll < len(phones.simple_transcript) and float(phones.simple_transcript[ll][1]) <= sylsend:
                    phonestart, phoneend, phone = phones.simple_transcript[ll]
                    phonestart = float(phonestart)
                    phoneend = float(phoneend)
                    ll += 1
                    if phonestart < sylsstart:
                        continue
                    syls_phones.append(phone)
                    durations.append(float(phoneend) - float(phonestart))
                if syls_phones:
                    word_syls.append(syls_phones)
                    durations.append(0.0)

            if word_syls == [] or len(durations) <= 1:
                continue
            durations.pop()  # remove last inserted 0.0

            if sum(durations) < 0.150:
                continue
            sampa = ' . '.join([' '.join(syl) for syl in word_syls])
            durations = ' '.join([f'{dur:.3f}' for dur in durations])
            # save later


            # save TextGrid
            sylstart_index = None
            while jj < len(syls.simple_transcript):
                sylstart, sylend, syl = syls.simple_transcript[jj]
                sylstart = float(sylstart)
                sylend = float(sylend)
                if not sylstart_index and sylstart >= wordstart:
                    sylstart_index = jj
                if sylend > wordend:
                    sylend_index = jj - 1
                    jj -= 1
                    break
                jj += 1

            syl_intervals = [(float(sylstart) - wordstart, float(sylend) - wordstart, str(kk)) for kk, (sylstart, sylend, syl) in enumerate(syls.simple_transcript[sylstart_index:(sylend_index + 1)])]
            if len(syl_intervals) == 0:
                print(f"{ii} - {word}: word with no syls")
                continue
            if len(syl_intervals) < 2:
                continue
                #dur = wordend - wordstart
                #syl_intervals = [(0.0, dur / 3, ""), (dur / 3, 2 / 3 * dur, "1"), (2 / 3 * dur, dur, "")]
            with open(f"{text_grid_dir}/{ii:06d}-{word}.TextGrid", 'wt') as tgfile:
                tgfile.write('File type = "ooTextFile"\n'
                             'Object class = "TextGrid"\n'
                             '\n'
                             '0\n'
                             f'{wordend - wordstart}\n'
                             '<exists>\n'
                             '2\n'
                             '"IntervalTier"\n'
                             '"Position"\n'
                             '0\n'
                             f'{wordend - wordstart}\n'
                             f'{len(syl_intervals)}\n')
                for sylstart, sylend, syl in syl_intervals:
                    tgfile.write(f'{sylstart}\n'
                                 f'{sylend}\n'
                                 f'"{syl}"\n')
                tgfile.write('"IntervalTier"\n'
                             '"ORT"\n'
                             '0\n'
                             f'{wordend - wordstart}\n'
                             '1\n'
                             '0\n'
                             f'{wordend - wordstart}\n'
                             f'"{word}"')

            # save wav
            wav_wordstart = floor(wordstart * samplerate)
            wav_wordend = floor(wordend * samplerate)

            wavfile.write(f"{wav_dir}/{ii:06d}-{word}.wav", samplerate, wav_data[wav_wordstart:wav_wordend + 1])

            # save utt
            utt_file.write(f'{word}\t/{sampa}/\t{durations}\n')

            ii += 1

    return ii

