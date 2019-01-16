from textgrid import TextGrid

tg_name = 'mono.A-C.left.textGrid'
utt_name = 'text.utt'

with open(tg_name, 'rt', encoding='UTF-16') as tg_file:
    fid = TextGrid(tg_file.read())

phones, syls, words = fid.tiers

with open(utt_name, 'wt') as utt_file:
    # write header
    utt_file.write('Label\tSampa transcription\tPhone durations\n')

    ii = 0
    jj = 0

    for wordstart, wordend, word in words.simple_transcript:

        wordstart = float(wordstart)
        wordend = float(wordend)

        if word in ('<P>', '<LAUGH>'):
            continue

        word_syls = []
        durations = []
        while ii < len(syls.simple_transcript) and float(syls.simple_transcript[ii][1]) <= wordend:
            sylsstart, sylsend, _ = syls.simple_transcript[ii]
            sylsstart = float(sylsstart)
            sylsend = float(sylsend)
            ii += 1
            if sylsstart < wordstart:
                continue
           
            syls_phones = []
            while jj < len(phones.simple_transcript) and float(phones.simple_transcript[jj][1]) <= sylsend:
                phonestart, phoneend, phone = phones.simple_transcript[jj]
                phonestart = float(phonestart)
                phoneend = float(phoneend)
                jj += 1
                if phonestart < sylsstart:
                    continue
                syls_phones.append(phone)
                durations.append(float(phoneend) - float(phonestart))
            word_syls.append(syls_phones)
            durations.append(0.0)

        if word_syls == [] or len(durations) <= 1:
            continue
        durations.pop()  # remove last inserted 0.0

        sampa = ' . '.join([' '.join(syl) for syl in word_syls])
        durations = ' '.join([f'{dur:.3f}' for dur in durations])
        utt_file.write(f'{word}\t/{sampa}/\t{durations}\n')

