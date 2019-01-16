import os

from .generate_gestural_score import generate_gestural_score

def sampa_to_ges(utt_name, ges_dir, *, phone_attributes):

    with open(utt_name, 'rt') as uttfile:

        uttfile.readline()  # skip header

        for ii, line in enumerate(uttfile):
            utterance, sampa, durations = line.split('\t')
            sampa = sampa.split()
            durations = durations.split()
            if durations == []:  # if no durations are given set to None
                durations = None
            else:
                assert len(sampa) == len(durations)
            if not sampa:
                raise ValueError("sampa transcription is missing")
            sampa[0] = sampa[0].strip('/')
            sampa[-1] = sampa[-1].strip('/')

            base_name = f'{ii:06d}-{utterance:.16s}'

            # generate gestural score file
            try:
                generate_gestural_score(f'{ges_dir}/{base_name}.ges', sampa, durations, phone_attributes=phone_attributes)
            except KeyError as e:
                print(e)
                continue
            except IndexError as e:
                print(base_name)
                raise e

            # WARNING: Because of carry over effects in VocalTractLabApi we have to
            # cleanly load it every time.
            # This seems only be safely possible by putting it in a seperate process.
            #failure = os.system(f'python ges2wav.py {base_name}')
            #if failure != 0:
            #    raise OSError(f'ges2wav failed with error code {failure}.')

