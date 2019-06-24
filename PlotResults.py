from pickle import load
from collections import OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy import stats
from statistics import mean
from ModelPrediction import low_pass_filter, high_pass_filter, binary_filter
from OfferStatistics import plot_data
from matplotlib import rcParams


DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'

EMOTIONS = ['happy', 'sad', 'neutral']

EPSILON = 0.0000001


def plot_key_by_emotions_list_diff(data, emotion_type_key, emotion_type_bl_key, emotion_list, y_keys):
    fig = plt.figure()
    for i,y_key in enumerate(y_keys):
        x = []
        y = []
        for part_data in data.values():
            s = 0
            for e in emotion_list:
                diff = part_data[emotion_type_key][e] - part_data[emotion_type_bl_key][e]
                s = s + diff
            x.append(s)
            y.append(part_data[y_key])
        ax = fig.add_subplot(len(y_keys), 1, i+1)
        x_label = 'sum of: {}: {}'.format(emotion_type_key + ' diff', emotion_list)
        plot_data(ax, x, y, x_label, y_key)
    plt.show()


def plot_key_by_emotions_list_ratio(data, emotion_type_key, emotion_type_bl_key, emotion_list, y_keys):
    fig = plt.figure()
    MAX = 5
    for i,y_key in enumerate(y_keys):
        x = []
        y = []
        for part_data in data.values():
            s = 0
            for e in emotion_list:
                if part_data[emotion_type_key][e] == 0:
                    ratio = 0
                elif part_data[emotion_type_bl_key][e] == 0.0:
                    ratio = MAX  # part_data[emotion_type_key][e] / 0.0000001
                else:
                    ratio = part_data[emotion_type_key][e] / part_data[emotion_type_bl_key][e]
                    if ratio > MAX:
                        continue
                s = s + ratio
            x.append(s)
            y.append(part_data[y_key])
        ax = fig.add_subplot(len(y_keys), 1, i+1)
        x_label = 'sum of: {}: {}'.format(emotion_type_key + ' ratio',
                                          emotion_list)
        plot_data(ax, x, y, x_label, y_key)
    plt.show()


def plot_key_by_emotions_list(data, emotion_type_key, emotion_list, y_keys):
    fig = plt.figure()
    for i,y_key in enumerate(y_keys):
        x = []
        y = []
        for part_data in data.values():
            s = 0
            for e in emotion_list:
                s = s + part_data[emotion_type_key][e]
            x.append(s)
            y.append(part_data[y_key])
        ax = fig.add_subplot(len(y_keys), 1, i+1)
        x_label = 'sum of: {}: {}'.format(emotion_type_key, emotion_list)
        plot_data(ax, x, y, x_label, y_key)
    plt.show()


def plot_all_emotions(data, emotion_type_key, emotion_list, y_keys,legend_keys):
    keys = {
        'ultimatumOffer': "Ultimatum Offer",
        'trustOffer': 'Trust Offer'
    }
    font_size = 14
    fig = plt.figure(figsize=(10,20))
    for j,list in enumerate(emotion_list):
        for i, y_key in enumerate(y_keys):
            x = []
            y = []
            for part_data in data.values():
                s = 0
                for e in list:
                    s = s + part_data[emotion_type_key][e]
                x.append(s)
                y.append(part_data[y_key])
            ax = fig.add_subplot(len(y_keys), 1, i+1)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            ax.plot(x, [intercept + slope * float(x1) for x1 in x],label= legend_keys[j] + ': y={:.2f}x+{:.2f}, p-value={:.2f}'.
                     format(slope, intercept, p_value, r_value ** 2))
            rcParams.update({'font.size': font_size})
            plt.ylabel(keys.get(y_key), fontsize=font_size)
            plt.xlabel('emotion strength as measured by self-report', fontsize=font_size)
            leg = ax.legend()
    plt.suptitle('Offer Sum vs Emotion Strength ')
    plt.show()


def plot_data_correlations(data,feature):
    trust = []
    ultimatum = []
    key_values = []
    feature_values = []
    for sample in data.values():
        if feature in sample:
            trust.append(sample['trustOffer'])
            ultimatum.append(sample['ultimatumOffer'])
            feature_values.append(sample[feature])

    font_size = 14
    ax = plt.subplot(2,1,1)
    ax.scatter(feature_values,trust)
    plt.ylabel('trust offer', fontsize=font_size)
    plt.xlabel(feature)
    # fit data
    slope, intercept, r_value, p_value, std_err = stats.linregress(feature_values, trust)
    # Note: Two-sided p-value for a hypothesis test whose null hypothesis is
    # that the slope is zero, using Wald Test with t-distribution of the
    # test statistic.
    ax.plot(feature_values, [intercept + slope * float(x1) for x1 in feature_values])
    plt.text(0.5, 0.8, 'y={:.2f}x+{:.2f}, p-value={:.2f}, r^2={:.2f}'.
             format(slope, intercept, p_value, r_value**2),
             horizontalalignment='center', verticalalignment='center',
             fontsize=font_size, transform=ax.transAxes)


    ax2 = plt.subplot(2,1,2)
    ax2.scatter(feature_values,ultimatum)
    plt.ylabel('ultimatum offer', fontsize=font_size)
    plt.xlabel(feature)
    # fit data
    slope, intercept, r_value, p_value, std_err = stats.linregress(feature_values, ultimatum)
    # Note: Two-sided p-value for a hypothesis test whose null hypothesis is
    # that the slope is zero, using Wald Test with t-distribution of the
    # test statistic.
    ax2.plot(feature_values, [intercept + slope * float(x1) for x1 in feature_values])
    plt.text(0.5, 0.8, 'y={:.2f}x+{:.2f}, p-value={:.2f}, r^2={:.2f}'.
             format(slope, intercept, p_value, r_value**2),
             horizontalalignment='center', verticalalignment='center',
             fontsize=font_size, transform=ax2.transAxes)
    plt.show()


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))
    print(len(data_sorted))
    # data_sorted = low_pass_filter(data_sorted, 'introversion', mean)
    #data_sorted = binary_filter(data_sorted, 'gender', 0)
    print(len(data_sorted))

    ultimatum = 'ultimatumOffer'
    trust = 'trustOffer'
    feature = 'yearOfBirth'

    sad_list = ['sadness', 'grief']
    happy_list = ['happiness', 'amusement']
    neutral_list = ['calm','apathy']
    all_emotions = [sad_list, happy_list, neutral_list]
    emotion_keys = ['sad', 'happy','neutral']

    # plot_key_by_emotions_list(data_sorted, 'selfReport', sad_list, [ultimatum, trust])

    plot_all_emotions(data_sorted, 'selfReport', all_emotions, [ultimatum, trust], emotion_keys)

    # plot_data_correlations(data_sorted,feature)

    # plot_key_by_emotions_list_ratio(data_sorted, 'videoMean', 'videoBLMean', ['sad'], [ultimatum, trust])

    # plot_key_by_emotions_list_diff(data_sorted, 'videoFreqTopPercent', 'videoBLFreqTopPercent', ['sad'], [ultimatum, trust])



if __name__ == "__main__":
    main()