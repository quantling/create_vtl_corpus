
from collections import namedtuple

Utterance = namedtuple('Utterance', ['sampa',
                                     'inner_trans',
                                     'gesture',
                                     'duration',
                                     'tao',
                                     'manner',
                                     'glottal',
                                     'place'])

STOPS = ('p', 't', 'k', 'b', 'd', 'g')

def generate_gestural_score(ges_name, sampa, durations=None, *, phone_attributes):

    tc_pho = 0.015  # specific time_constant of regular gestures
    tc_lp = 0.005  # specific time_constant of lung-pressure gestures
    tc_lp_beg = 0.005  # specific duration of the beginning lung-pressure gesture
    tc_lp_end = 0.015  # specific duration of the ending lung-pressure gesture
    empty_lp_dur_beg = 0.01  #% in this new fashion, it is re-written based on the type of the initial consonant
    empty_lp_dur_end = 0.03
    lp_value = 8000  # specific value of valid lung-pressure gesture in [dPa]
    f0_value = 80  # specific value of F0 in [Hz]

    phone_mapping = dict()
    with open(phone_attributes, 'rt') as map_file:
        for line in map_file:
            key, *values = line.split()
            if len(values) != 7:
                raise ValueError("Each phone needs 7 attributes, i. e. 8 coloumns")
            phone_mapping[key] = tuple(values)

    inner_tran = []
    phone_count = 0
    phone_num = len([phone for phone in sampa if phone != '.'])  # excluding delimiter '.'

    utterance = Utterance([], [], [], [], [], [], [], [])

    for ii, phone in enumerate(sampa):
        if phone in ('', '.'):
            continue
        utterance.sampa.append(phone)
        utterance.inner_trans.append(phone_mapping[phone][0])
        utterance.gesture.append(phone_mapping[phone][1])
        if durations is None:
            utterance.duration.append(float(phone_mapping[phone][2]))
        else:
            utterance.duration.append(float(durations[ii]))
        utterance.tao.append(float(phone_mapping[phone][3]))
        utterance.manner.append(phone_mapping[phone][4])
        utterance.glottal.append(phone_mapping[phone][5])
        utterance.place.append(phone_mapping[phone][6])


    ##########################################################################
    # write gestural score file
    ##########################################################################

    with open(ges_name, 'wt') as ges_file:
        ges_file.write('<gestural_score>\n')


        # write vowel tier
        ###########################################################################
        ges_file.write('  <gesture_sequence type="vowel-gestures" unit="">\n')
        #ges_file.write('    <gesture value="" slope="0.000000" duration_s="0.030000" time_constant_s="0.015000" neutral="1" />\n')  # phase in


        # list of syllables where each syllable is a list of phones
        syllables = [syllable.split() for syllable in ' '.join(sampa).split(' . ')]

        phone_count = 0

        for syllable in syllables:
            vowel_flag = False
            vowel_duration = 0
            preceeding_tao = 0.015
            syllable_duration = 0

            for phone in syllable:
                phone_duration = utterance.duration[phone_count]

                if utterance.manner[phone_count] == 'vowel':
                    vowel_flag = True
                    if utterance.sampa[phone_count] == '6':  # openSchwa
                        if phone_count > 0 and utterance.manner[phone_count - 1] == 'vowel':  # the previous phone is a vowel
                            # here, do not add the duration of previous vowel
                            # also, use left-context dependent open-schwa
                            open_schwa_gesture = '@6-to-@6'
                            #if utterance.gesture[phone_count - 1] in ('i','I'):
                            #    open_schwa_gesture = '@6-to-[iI]@6'
                            #elif utterance.gesture[phone_count - 1] in ('u','U'):
                            #    open_schwa_gesture = '@6-to-[uU]@6'
                            #elif utterance.gesture[phone_count - 1] in ('o','O'):
                            #    open_schwa_gesture = '@6-to-[oO]@6'
                            #else:
                            #    open_schwa_gesture = '@6-to-@6'

                            return_str = write_one_gesture(open_schwa_gesture, phone_duration, preceeding_tao, 0)
                            ges_file.write(return_str)

                            syllable_duration += phone_duration
                            vowel_duration += phone_duration
                        else:  # this phone is the central vowel
                            syllable_duration += phone_duration
                            vowel_duration += phone_duration

                            return_str = write_one_gesture(utterance.gesture[phone_count], phone_duration, preceeding_tao, 0)
                            ges_file.write(return_str)
                    else:  # this phone is a regular vowel, not open-schwa
                        # when it is a diphthong, split it into beginning part
                        # (1/3 duration) and ending part (2/3 duration). And
                        # the beginning part and the preceding consonant will be merged.
                        if phone in ('aI', 'aU', 'OY', 'i:6', 'O6', 'e:6',
                                'I6', 'E6', 'E:6', 'eI', 'y:6', 'a:6', 'u:6',
                                'Y6', 'U6', 'o:6', 'a6', '2:6'):
                            if len(phone) == 2:
                                diphthong_beg = phone[0]
                                diphthong_end = phone[1]
                            elif phone == 'e:6':
                                diphthong_beg = 'e'
                                diphthong_end = '6'
                            elif phone == 'y:6':
                                diphthong_beg = 'y'
                                diphthong_end = '6'
                            elif phone == '2:6':
                                diphthong_beg = '2'
                                diphthong_end = '6'
                            elif len(phone) == 3:
                                diphthong_beg = phone[0:2]
                                diphthong_end = phone[2]

                            diphthong_beg_dur = 1/3 * phone_duration
                            diphthong_end_dur = 2/3 * phone_duration
                            # the operation ensures that the 1/3 boundary point occurs exactly
                            # at 1/3 point of this diphthong phone, at not the diphthong gesture
                            return_str = write_one_gesture(diphthong_beg, vowel_duration + diphthong_beg_dur, 0.010, 0)
                            return_str += write_one_gesture(diphthong_end, diphthong_end_dur, 0.015, 0)
                            ges_file.write(return_str)

                            syllable_duration += phone_duration
                            vowel_duration += phone_duration
                        else:
                            syllable_duration += phone_duration
                            vowel_duration += phone_duration
                            return_str = write_one_gesture(utterance.gesture[phone_count], vowel_duration, preceeding_tao, 0)
                            ges_file.write(return_str)
                else:  #  this phone is a consonant, then just calculate duration
                    syllable_duration += phone_duration
                    # save tao so as to initialize the time-constant of central vowel
                    preceeding_tao = utterance.tao[phone_count]
                    if not vowel_flag:
                        vowel_duration += phone_duration

                phone_count += 1

            # if this is a closed-syllable, then a neutral gesture(time_constant==15ms, neutral==1) at vowel
            # tier is imposed, which is above the final consonant(s) of the
            # current syllable
            if syllable_duration > vowel_duration:
                return_str = write_one_gesture('', syllable_duration - vowel_duration, 0.015, 1)
                ges_file.write(return_str)
            elif syllable_duration < vowel_duration:
                raise ValueError("syllable_duration should never be strictly smaller than vowel duration")


        #innerTrans=regexprep(innerTrans, '\.', '');
        #phoneCell=regexp(innerTrans, '(\S+)', 'tokens');

        # judge whether the last phoneme is a stop
        # if true, then add an empty gesture(duration==30ms, time_constant==5ms,
        # neutral==0) at vowel tier, which ensures the stop release
        if utterance.sampa[-1] in STOPS:
            return_str = write_one_gesture('', 0.030, 0.005, 0)
            ges_file.write(return_str)
        ges_file.write('  </gesture_sequence>\n')


        # write lip tier
        ###########################################################################
        ges_file.write('  <gesture_sequence type="lip-gestures" unit="">\n')
        #ges_file.write('    <gesture value="" slope="0.000000" duration_s="0.030000" time_constant_s="0.015000" neutral="1" />\n')  # phase in

        return_str, total_duration = write_reg_con_tier(utterance, 'lip')  # the last parameter indicates the palce of articulation
        ges_file.write(return_str)
        if return_str == '':
            ges_file.write(f'    <gesture value="" slope="0.000000" duration_s="{total_duration:.6f}" time_constant_s="0.015000" neutral="1" />\n')
        ges_file.write('  </gesture_sequence>\n')


        # write tongue-tip tier
        ###########################################################################
        ges_file.write('  <gesture_sequence type="tongue-tip-gestures" unit="">\n')
        #ges_file.write('    <gesture value="" slope="0.000000" duration_s="0.030000" time_constant_s="0.015000" neutral="1" />\n')  # phase in
        return_str, total_duration = write_reg_con_tier(utterance, 'tongue-tip')  # the last parameter indicates the palce of articulation
        ges_file.write(return_str)
        if return_str == '':
            ges_file.write(f'    <gesture value="" slope="0.000000" duration_s="{total_duration:.6f}" time_constant_s="0.015000" neutral="1" />\n')
        ges_file.write('  </gesture_sequence>\n')


        # write tongue-body tier
        ###########################################################################
        ges_file.write('  <gesture_sequence type="tongue-body-gestures" unit="">\n')
        #ges_file.write('    <gesture value="" slope="0.000000" duration_s="0.030000" time_constant_s="0.015000" neutral="1" />\n')  # phase in
        return_str, total_duration = write_reg_con_tier(utterance, 'tongue-body')  # the last parameter indicates the palce of articulation
        ges_file.write(return_str)
        if return_str == '':
            ges_file.write(f'    <gesture value="" slope="0.000000" duration_s="{total_duration:.6f}" time_constant_s="0.015000" neutral="1" />\n')
        ges_file.write('  </gesture_sequence>\n')


        # write velic tier
        ###########################################################################
        # just filling an empty gesture in case of aborting in API call,
        # but this is not mandatory for initializing purpose
        ges_file.write('  <gesture_sequence type="velic-gestures" unit="">\n')
        ges_file.write(f'    <gesture value="0.500000" slope="0.000000" duration_s="0.010000" time_constant_s="0.015000" neutral="1" />\n')
        ges_file.write('  </gesture_sequence>\n')


        # write glottal-shape tier
        ###########################################################################
        ges_file.write('  <gesture_sequence type="glottal-shape-gestures" unit="">\n')
        #ges_file.write('    <gesture value="" slope="0.000000" duration_s="0.030000" time_constant_s="0.015000" neutral="1" />\n')  # phase in
        return_str = write_glottal_tier(utterance, 0.015)  # the last parameter is tao (time constant)
        # if true, then add an empty gesture(duration==30ms, time_constant==5ms,
        # neutral==0), which ensures the stop release
        if utterance.sampa[-1] in STOPS:
            temp_str= '    <gesture value="" slope="0.000000" duration_s="0.030000" time_constant_s="0.010000" neutral="0" />\n'
            return_str += temp_str
        ges_file.write(return_str)
        ges_file.write('  </gesture_sequence>\n')


        # write F0 tier
        ###########################################################################
        # the following lines give the real duration of this utterance
        # considering the lung-pressure tier, the valid gesture starts very
        # after the first phone. So its duration does not account the first
        # phone. But here add 50ms, potentially help the relase of the last
        # (stop) phone

        totoal_duration = sum(utterance.duration)

        # judge whether the last phoneme is a stop
        # if true, then add an empty gesture(duration==30ms, time_constant==5ms) at vowel tier,
        # and an extra duration (30ms) at then end of valid lung-pressure gesture, which ensures the stop release
        if utterance.sampa[-1] in STOPS:
            total_duration += 0.030
        ges_file.write('  <gesture_sequence type="f0-gestures" unit="st">\n')
        #ges_file.write(f'    <gesture value="{f0_value:.6f}" slope="0.000000" duration_s="0.030000" time_constant_s="0.015000" neutral="1" />\n')  # phase in
        return_str = write_constant_tier(total_duration, f0_value, tc_pho)
        ges_file.write(return_str)
        ges_file.write('  </gesture_sequence>\n')


        # write lung-pressure tier
        ###########################################################################
        # at this tier, it starts with an empty lung-pressure gesture and lasts
        # 100%, one-half or one-third of the initial consonant for stop,
        # fricative or sonorants, respectively. This is an empirical
        # implementation as so to make the word-initial sound correct.

        #initial_constant_dur = utterance.duration[0]
        #if '-stop' in utterance.gesture[0]:  # for stops, the left boundary of the first valid lung-pressure gesture
        #    first_lung_pressure_dur = initial_constant_dur
        #elif '-fric' in utterance.gesture[0]:
        #    first_lung_pressure_dur = 0.5 * initial_constant_dur
        #elif any(ending in utterance.gesture[0] for ending in ('-nas', '-lat', '-null')):  # sonorants or glottals ( e.g. 'h' or '?')
        #    first_lung_pressure_dur = 1 / 3 * initial_constant_dur
        #else:
        #    first_lung_pressure_dur = 0.01

        #ges_file.write('  <gesture_sequence type="lung-pressure-gestures" unit="dPa">\n')
        #ges_file.write(f'    <gesture value="0" slope="0.000000" duration_s="{first_lung_pressure_dur:.3f}" time_constant_s="{tc_lp:.3f}" neutral="0" />\n')

        ges_file.write('  <gesture_sequence type="lung-pressure-gestures" unit="dPa">\n')
        #ges_file.write(f'    <gesture value="0.000000" slope="0.000000" duration_s="0.010000" time_constant_s="{tc_lp:.6f}" neutral="0" />\n')  # phase in 1/3
        #ges_file.write(f'    <gesture value="{lp_value/3:.6f}" slope="0.000000" duration_s="0.010000" time_constant_s="{tc_lp:.6f}" neutral="0" />\n')  # phase in 2/3
        #ges_file.write(f'    <gesture value="{lp_value*2/3:.6f}" slope="0.000000" duration_s="0.010000" time_constant_s="{tc_lp:.6f}" neutral="0" />\n')  # phase in 3/3
        #second_lung_pressure_dur = total_duration - first_lung_pressure_dur
        second_lung_pressure_dur = total_duration
        if second_lung_pressure_dur > 0.0:
            return_str = write_constant_tier(second_lung_pressure_dur, lp_value, tc_lp)
            ges_file.write(return_str)
        ges_file.write(f'    <gesture value="0.000000" slope="0.000000" duration_s="{empty_lp_dur_end:.6f}" time_constant_s="{tc_lp:.6f}" neutral="0" />\n')
        ges_file.write('  </gesture_sequence>\n')

        ges_file.write('</gestural_score>')


def write_one_gesture(phone, duration, time_constant, neutral_flag):
    """
    write one pseudo gesture when this tier does not have a real valid
    gesture, which is used to conveniently read and write by xml toolbox

    """
    return f'    <gesture value="{phone}" slope="0.000000" duration_s="{duration:.6f}" time_constant_s="{time_constant:.6f}" neutral="{neutral_flag:d}" />\n'


def print_empty_gesture(empty_dur):
    temp_str=''
    while empty_dur > 1:
        temp_dur = 0.5
        temp_str += f'  	<gesture value="" slope="0.000000" duration_s="{temp_dur:.6f}" time_constant_s="0.015000" neutral="1" />\n'
        empty_dur -= temp_dur
    if empty_dur < 0.01:
        empty_dur = 0.01
    temp_str += f'  	<gesture value="" slope="0.000000" duration_s="{empty_dur:.6f}" time_constant_s="0.015000" neutral="1" />\n'
    return temp_str


def write_constant_tier(total_duration, default_value, tc):
    """
    this function returns a string gestures of constant tiers (e.g. F0 and lung-pressure)

    defaultValue: default value of this constant tier
    tc: time_constant

    """

    return_str = ''

    # if it is a long empty gesture,
    # then split it into multiple gestures
    while total_duration > 1:
        temp_dur = 0.5
        return_str += f'    <gesture value="{default_value:.6f}" slope="0.000000" duration_s="{temp_dur:.6f}" time_constant_s="{tc:.6f}" neutral="0" />\n'
        total_duration -= temp_dur

    if total_duration < 0.01:
        total_duration=0.01
    return_str += f'    <gesture value="{default_value:.6f}" slope="0.000000" duration_s="{total_duration:.6f}" time_constant_s="{tc:.6f}" neutral="0" />\n'
    return return_str


def write_glottal_tier(utterance, tao):
    return_str = ''
    glottal_dur = utterance.duration
    for ii in range(len(utterance.duration)):
        return_str += f'    <gesture value="{utterance.glottal[ii]}" slope="0.000000" duration_s="{utterance.duration[ii]:.6f}" time_constant_s="{tao:.6f}" neutral="0" />\n'
    return return_str


def write_reg_con_tier(utterance, arti_place):
    """
    this function returns a string containing gestures of the current regular consonant tier

    utterStruct: struct of the utterance
    artiPlace: articulation place

    """

    total_dur = 0
    return_str = ''

    for ii in range(len(utterance.duration)):
        total_dur += utterance.duration[ii]
        # Is this phone valid at the current tier?
        if utterance.place[ii] == arti_place:
            return_str += write_one_gesture(utterance.gesture[ii], utterance.duration[ii], utterance.tao[ii], 0)
        # this phone does not belongs to current tier print an empty gesture
        else:
            return_str += print_empty_gesture(utterance.duration[ii])

    # judge whethe the last phoneme is a stop
    # if true, then add an empty gesture(duration==30ms, time_constant==5ms,
    # neutral==0), which ensures the stop release
    if utterance.sampa[-1] in STOPS:
        return_str += write_one_gesture('', 0.030, 0.005, 0)
        total_dur += 0.030

    return (return_str, total_dur)

