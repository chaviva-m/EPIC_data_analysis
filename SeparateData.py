from pickle import load
from collections import OrderedDict, defaultdict
from operator import itemgetter
import numpy as np
from statistics import mean, stdev
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tabulate import tabulate
from itertools import combinations
from OfferStatistics import stats_offer_by_param
from ModelPrediction import low_pass_filter, high_pass_filter

DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'

EPSILON = 0.0000001


def stats_by_emotion(all_data, key):
    print(key)
    mean_key = dict()
    std_key = dict()
    key_all_lists = dict()
    # mean & std
    for name,data in all_data.items():
        key_all_values = [part_data[key] for part_data in data.values()]
        key_all_lists[name] = key_all_values
        mean_key[name] = mean(key_all_values)
        std_key[name] = stdev(key_all_values) if len(key_all_values) > 1 else float('inf')
        print('{}: mean = {:0.3f}, std = {:0.3f}, n = {}'.format(name, mean_key[name],
                                                     std_key[name], len(data)))
    # t-test
    print('t-test')
    for pair in combinations(key_all_lists.items(), 2):
        if len(pair[0][1]) <=1 or len(pair[1][1]) <= 1:
            continue
        [t_stat, p] = ttest_ind(pair[0][1], pair[1][1], equal_var=False)
        print('{} vs {}: t-stat={:.3f}, p-val={:.3f}'.format(pair[0][0], pair[1][0],t_stat, p))
    return mean_key, std_key


def print_datasets(all_data, separation_type):
    print('')
    print(separation_type)
    print('data separation:')
    for emotion,data in all_data.items():
        print(emotion, end=': ')
        for part_id, part_data in data.items():
            print(part_id, end=' ')
        print('')
    print('')


def separate_by_self_report1(data):
    emotion_names = ['happy', 'sad', 'neutral']
    data_separated = defaultdict(dict)
    for part_id,part_data in data.items():
        emotions = part_data['selfReport']
        happy = emotions['happiness'] + emotions['amusement']
        sad = emotions['sadness'] + emotions['grief']
        neutral = emotions['calm'] + emotions['apathy']
        max_emotion = np.argmax([happy, sad, neutral])
        data_separated[emotion_names[max_emotion]][part_id] = part_data
    return data_separated


def separate_by_self_report2(data):
    data_separated = defaultdict(dict)
    for part_id,part_data in data.items():
        emotions = part_data['selfReport']
        happy = emotions['happiness'] + emotions['amusement']
        sad = emotions['sadness'] + emotions['grief']
        neutral = emotions['calm'] + emotions['apathy']
        if happy > max(sad, neutral) + 1:
            data_separated['happy'][part_id] = part_data
        elif sad > max(happy, neutral) + 1:
            data_separated['sad'][part_id] = part_data
        else:
            data_separated['neutral'][part_id] = part_data
    return data_separated


def separate_by_self_report3(data):
    threshold = 8
    data_separated = defaultdict(dict)
    for part_id,part_data in data.items():
        emotions = part_data['selfReport']
        happy = emotions['happiness'] + emotions['amusement']
        sad = emotions['sadness'] + emotions['grief']
        neutral = emotions['calm'] + emotions['apathy']
        if happy > max(sad, neutral) and happy > threshold:
            data_separated['happy'][part_id] = part_data
        elif sad > max(happy, neutral) and sad > threshold:
            data_separated['sad'][part_id] = part_data
        else:
            data_separated['neutral'][part_id] = part_data
    return data_separated


def separate_by_recording(data, recording_key):
    data_separated = defaultdict(dict)
    emotion_names = ['happy', 'sad', 'neutral']
    for part_id,part_data in data.items():
        emotions = part_data[recording_key]
        emotion_strengths = []
        for name in emotion_names:
            strength = emotions[name]
            emotion_strengths.append(strength)
        max_emotion = np.argmax(emotion_strengths)
        data_separated[emotion_names[max_emotion]][part_id] = part_data
    return data_separated


def separate_by_recording_ratio(data, recording_key, recording_bl_key):
    data_separated = defaultdict(dict)
    emotion_names = ['happy', 'sad', 'neutral']
    for part_id,part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recording_bl_key]
        emotion_strengths = []
        for name in emotion_names:
            emotion = emotions[name]
            emotion_bl = emotions_bl[name]
            if emotion_bl == 0:
                emotion_bl += EPSILON
            ratio = emotion / emotion_bl
            emotion_strengths.append(ratio)
        max_emotion = np.argmax(emotion_strengths)
        data_separated[emotion_names[max_emotion]][part_id] = part_data
    return data_separated


def separate_by_recording_diff(data, recording_key, recording_bl_key):
    data_separated = defaultdict(dict)
    emotion_names = ['happy', 'sad', 'neutral']
    for part_id,part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recording_bl_key]
        emotion_strengths = []
        for name in emotion_names:
            emotion = emotions[name]
            emotion_bl = emotions_bl[name]
            emotion_diff = emotion - emotion_bl
            emotion_strengths.append(emotion_diff)
        max_emotion = np.argmax(emotion_strengths)
        data_separated[emotion_names[max_emotion]][part_id] = part_data
    return data_separated


def plot_means(means_all, colors, data_titles, y_label):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    font_size = 14
    rcParams.update({'font.size': font_size})
    for i,means in enumerate(means_all):
        mean_sorted = OrderedDict(sorted(means.items(), key=itemgetter(0)))
        # std_sorted = OrderedDict(sorted(std_key.items(), key=itemgetter(0)))
        ax.plot(mean_sorted.keys(), mean_sorted.values(), 'o-', color=colors[i])
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.legend(data_titles, fontsize=font_size-2)
    ax.set_title('mean {}'.format(y_label))


def plot_means_all_separation_types(data, key):
    print(key)
    all_data = separate_by_self_report1(data)
    print_datasets(all_data, 'self report 1')
    m1, s1 = stats_by_emotion(all_data, key)
    all_data = separate_by_recording(data, 'videoFreqTopPercent')
    print_datasets(all_data, 'video freq top percent')
    m2, s2 = stats_by_emotion(all_data, key)
    all_data = separate_by_recording_diff(data, 'videoFreqTopPercent', 'videoBLFreqTopPercent')
    print_datasets(all_data, 'video freq top percent diff')
    m3, s3 = stats_by_emotion(all_data, key)
    all_data = separate_by_recording(data, 'audio')
    print_datasets(all_data, 'audio')
    m4, s4 = stats_by_emotion(all_data, key)
    all_data = separate_by_recording_diff(data, 'audio', 'audioBL')
    print_datasets(all_data, 'audio diff')
    m5, s5 = stats_by_emotion(all_data, key)
    plot_means([m1, m2, m3, m4, m5], ['k', 'r', 'g', 'c', 'm'],
               ['self report', 'video freq top %', 'video freq top % diff',
                'audio', 'audio diff'], key)


def evaluate_separation(data_sep, data_sep_gold):
    precision = dict()
    recall = dict()
    F1 = dict()
    correct_participants = defaultdict(list)
    for emotion,data in data_sep.items():
        participants = data.keys()
        participants_gold = data_sep_gold[emotion].keys()
        c = 0
        for p in participants:
            if p in participants_gold:
                c += 1
                correct_participants[emotion].append(p)
        precision[emotion] = c / len(participants)
        recall[emotion] = c / len(participants_gold)
        if precision[emotion] == 0 or recall[emotion] == 0:
            F1[emotion] = 0
        else:
            F1[emotion] = 2 * ((precision[emotion] * recall[emotion])/
                           (precision[emotion] + recall[emotion]))
    return [correct_participants, precision, recall, F1]


def add_separation_type_to_table(table_rows, feature, results):
    table_rows += [[feature, 'happy'] + [v['happy'] for v in results],
                      ['', 'sad'] + [v['sad'] for v in results],
                      ['', 'neutral'] + [v['neutral'] for v in results]]


def compare_separations(data_sorted):

    gold = separate_by_self_report1(data_sorted)

    table_rows = []

    # video ratio
    videoFreqRatio = separate_by_recording_ratio(data_sorted,'videoFreq',
                                                 'videoBLFreq')
    vfr = evaluate_separation(videoFreqRatio, gold)
    add_separation_type_to_table(table_rows, 'video freq ratio', vfr)

    videoThresholdFreqRatio = separate_by_recording_ratio\
        (data_sorted, 'videoThresholdFreq', 'videoBLThresholdFreq')
    vtfr = evaluate_separation(videoThresholdFreqRatio, gold)
    add_separation_type_to_table(table_rows, 'video threshold freq ratio', vtfr)

    videoFreqSkipBegRatio = separate_by_recording_ratio\
        (data_sorted, 'videoFreqSkipBeg', 'videoBLFreqSkipBeg')
    vfsbr = evaluate_separation(videoFreqSkipBegRatio, gold)
    add_separation_type_to_table(table_rows, 'video freq skip beginning ratio', vfsbr)

    videoFreqTopPercentRatio = separate_by_recording_ratio\
        (data_sorted, 'videoFreqTopPercent', 'videoBLFreqTopPercent')
    vftpr = evaluate_separation(videoFreqTopPercentRatio, gold)
    add_separation_type_to_table(table_rows, 'video freq top % ratio', vftpr)

    videoMeanRatio = separate_by_recording_ratio(data_sorted, 'videoProbMean',
                                                 'videoBLProbMean')
    vmr = evaluate_separation(videoMeanRatio, gold)
    add_separation_type_to_table(table_rows, 'video mean prob ratio', vmr)

    # video diff
    videoFreqDiff = separate_by_recording_diff(data_sorted,'videoFreq',
                                               'videoBLFreq')
    vfd = evaluate_separation(videoFreqDiff, gold)
    add_separation_type_to_table(table_rows, 'video freq diff', vfd)

    videoThresholdFreqDiff = separate_by_recording_diff\
        (data_sorted, 'videoThresholdFreq', 'videoBLThresholdFreq')
    vtfd = evaluate_separation(videoThresholdFreqDiff, gold)
    add_separation_type_to_table(table_rows, 'video threshold freq diff', vtfd)

    videoFreqSkipBegDiff = separate_by_recording_diff\
        (data_sorted, 'videoFreqSkipBeg', 'videoBLFreqSkipBeg')
    vfsbd = evaluate_separation(videoFreqSkipBegDiff, gold)
    add_separation_type_to_table(table_rows, 'video freq skip beginning diff', vfsbd)

    videoFreqTopPercentDiff = separate_by_recording_diff\
        (data_sorted, 'videoFreqTopPercent', 'videoBLFreqTopPercent')
    vftpd = evaluate_separation(videoFreqTopPercentDiff, gold)
    add_separation_type_to_table(table_rows, 'video freq top % diff', vftpd)

    videoMeanDiff = separate_by_recording_diff(data_sorted, 'videoProbMean',
                                                 'videoBLProbMean')
    vmd = evaluate_separation(videoMeanDiff, gold)
    add_separation_type_to_table(table_rows, 'video mean prob diff', vmd)

    # audio
    audioRatio = separate_by_recording_ratio(data_sorted, 'audio',
                                                 'audioBL')
    ar = evaluate_separation(audioRatio, gold)
    add_separation_type_to_table(table_rows, 'audio ratio', ar)

    audioDiff = separate_by_recording_diff(data_sorted,'audio',
                                                 'audioBL')
    ad = evaluate_separation(audioDiff, gold)
    add_separation_type_to_table(table_rows, 'audio diff', ad)

    # print results in table
    t = tabulate(table_rows, headers=['feature', 'emotion', 'correct participants',
                                 'precision', 'recall', 'F1'], tablefmt='orgtbl')

    print(t)


def compare_separations_top_features(data_sorted):

    gold = separate_by_self_report1(data_sorted)

    table_rows = []

    # video
    videoFreqTopPercent = separate_by_recording(data_sorted,
                                                'videoFreqTopPercent')
    vftp = evaluate_separation(videoFreqTopPercent, gold)
    add_separation_type_to_table(table_rows, 'video freq top %', vftp)

    videoFreqTopPercentDiff = separate_by_recording_diff\
        (data_sorted, 'videoFreqTopPercent', 'videoBLFreqTopPercent')
    vftpd = evaluate_separation(videoFreqTopPercentDiff, gold)
    add_separation_type_to_table(table_rows, 'video freq top % diff', vftpd)

    # audio
    audio = separate_by_recording(data_sorted,'audio')
    a = evaluate_separation(audio, gold)
    add_separation_type_to_table(table_rows, 'audio', a)

    audioDiff = separate_by_recording_diff(data_sorted,'audio',
                                                 'audioBL')
    ad = evaluate_separation(audioDiff, gold)
    add_separation_type_to_table(table_rows, 'audio diff', ad)

    # print results in table
    t = tabulate(table_rows, headers=['feature', 'emotion', 'correct participants',
                                 'precision', 'recall', 'F1'], tablefmt='orgtbl')

    print(t)


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))

    ''' We can compare separation evaluations with filtered and unfiltered data '''
    data_sorted = high_pass_filter(data_sorted, 'introversion', mean)   # with high pass - no one comes out happy

    # compare_separations(data_sorted)

    compare_separations_top_features(data_sorted)

    print('')
    key = 'ultimatumOffer'
    plot_means_all_separation_types(data_sorted, key)

    key = 'trustOffer'
    plot_means_all_separation_types(data_sorted, key)

    plt.show()


if __name__ == "__main__":
    main()