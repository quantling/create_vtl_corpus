import os

def sampa_to_seg(utt_name, seg_dir):

    with open(utt_name, 'rt') as uttfile:

        uttfile.readline()  # skip header

        for ii, line in enumerate(uttfile):
            utterance, sampa, durations = line.split('\t')
            sampa = sampa.strip('/')
            sampa = sampa.split()
            durations = durations.split()
            if durations == []:  # if no durations are given set to None
                durations = None
            else:
                assert len(sampa) == len(durations), f"line {ii + 2}"
            if not sampa:
                raise ValueError(f"sampa transcription is missing in line {ii + 2}")

            base_name = f'{ii:06d}-{utterance:.16s}'

            try:
                write_seg(f'{seg_dir}/{base_name}.seg', sampa, durations)
            except KeyError as e:
                print(e)
                continue
            except IndexError as e:
                print(base_name)
                raise e


def write_seg(segment_file_name, sampa, durations):
    with open(segment_file_name, 'wt') as seg_file:
        seg_file.write('name = ; duration_s = 0.05000; \r\n')
        for phone, duration in zip(sampa, durations):
            duration = float(duration)
            if phone == '.':
                continue
            seg_file.write(f'name = {phone}; duration_s = {duration:.6f}; \r\n')
        seg_file.write('name = ; duration_s = 0.05000; \r\n')

