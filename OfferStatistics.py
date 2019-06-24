"""
Take statistics of offer according to variable groupings
statistics include:
for categorical: mean, std and t-test
for continuous: slope, intercept, r_value, p_value, std_err
"""


from pickle import load
from collections import OrderedDict
from operator import itemgetter
from statistics import mean, stdev, median
from scipy.stats import ttest_ind, linregress
from itertools import combinations
import matplotlib.pyplot as plt
import copy


DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'


def stats_offer_by_param(data, offer_key, param_key):
    """
    :param data: data dictionary
    :param offer_key: string: 'ultimatumOffer' or 'trustOffer'
    :param param_key: string: key for calculating offer statistics (e.g. 'gender')
    :return: dictionary: value of param: list of offers
    """
    print('{}: {}'.format(offer_key, param_key))
    all_param_values = set(part_data[param_key] for part_data in data.values())
    offers_all_param_values = {}
    # calculate stats of each param value
    for param_value in all_param_values:
        offers = [part_data[offer_key] for part_data in data.values()
                  if part_data[param_key] == param_value]
        print('{}: mean={:.3f}, std={:.3f}, n={}'.format(param_value, mean(offers),
              stdev(offers) if len(offers) > 1 else float('inf'), len(offers)))
        offers_all_param_values[param_value] = offers
    # calculate t-test between param values
    for pair in combinations(offers_all_param_values.items(), 2):
        if len(pair[0][1]) <=1 or len(pair[1][1]) <= 1:
            continue
        [t_stat, p] = ttest_ind(pair[0][1], pair[1][1], equal_var=False)
        print('{} vs {}: t-stat={:.3f}, p-val={:.3f}'.format(pair[0][0], pair[1][0],t_stat, p))
    return offers_all_param_values


def param_binary_groups(data, param, value):
    """
    Make multiclass variable binary
    :param data: data dictionary
    :param param: string: key to make binary (key should be multiclass variable)
    :param value: string: binary groups will be 1 (value) or 0 (other)
    :return: data dictionary, with param as binary
    """
    all_data_binary_param = copy.deepcopy(data)
    for part,part_data in data.items():
        if part_data[param] == value:
            all_data_binary_param[part][param] = 1  # value
        else:
            all_data_binary_param[part][param] = 0  # 'other'
    return all_data_binary_param


def param_groups_median_threshold(data, param):
    """
    Make continuous variable categorical:
    split data into 3 groups: 2 (high), 1 (median) and 0 (low)
    :param data: data dictionary
    :param param: key to make categorical (key should be continuous variable)
    :return: data dictionary, with param as categorical
    """
    all_values = [part_data[param] for part_data in data.values()]
    threshold = median(all_values)
    all_data_categorical_param = copy.deepcopy(data)
    for part,part_data in data.items():
        if part_data[param] > threshold:
            all_data_categorical_param[part][param] = 2 # 'high'
        elif part_data[param] == threshold:
            all_data_categorical_param[part][param] = 1 # 'median'
        else:
            all_data_categorical_param[part][param] = 0 # 'low'

    return all_data_categorical_param


def plot_keys_vs_param(data, param, y_keys):
    '''
    fit and plot regression line for continuous variable
    :param data: data dictionary
    :param param: key of continuous variable
    :param y_keys: list of keys to plot against param (e.g. ['ultimatumOffer', 'trustOffer'])
    :return: list of [slope, intercept, r_value, p_value, std_err] for each y_key
    '''
    fig = plt.figure()
    results = []
    for i,y_key in enumerate(y_keys):
        x = [part_data[param] for part_data in data.values()]
        y = [part_data[y_key] for part_data in data.values()]
        ax = fig.add_subplot(len(y_keys), 1, i+1)
        x_label = param
        [slope, intercept, r_value, p_value, std_err] = \
            plot_data(ax, x, y, x_label, y_key)
        results.append([slope, intercept, r_value, p_value, std_err])
    plt.show()
    return results


def plot_data(ax, x, y, x_label, y_label):
    '''
    fit and plot regression line
    :param ax: plot handle
    :param x: list of x data
    :param y: list of y data
    :param x_label: label of x axis
    :param y_label: label of y axis
    :return: [slope, intercept, r_value, p_value, std_err]
    '''
    font_size = 14
    ax.scatter(x,y)
    plt.ylabel(y_label, fontsize=font_size)
    plt.ylabel('trust offer', fontsize=font_size)
    plt.xlabel(x_label, fontsize=font_size)
    plt.xlabel('ultimatum offer', fontsize=font_size)
    # fit data
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # Note: Two-sided p-value for a hypothesis test whose null hypothesis is
    # that the slope is zero, using Wald Test with t-distribution of the
    # test statistic.
    ax.plot(x, [intercept + slope * float(x1) for x1 in x])
    plt.title('trust offer vs. ultimatum offer', fontsize=font_size)
    plt.text(0.5, 0.9, 'y={:.3f}x+{:.3f}, p-value={:.3f}, r^2={:.3f}'.
             format(slope, intercept, p_value, r_value**2),
             horizontalalignment='center', verticalalignment='center',
             fontsize=font_size, transform=ax.transAxes)
    return [slope, intercept, r_value, p_value, std_err]


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))

    # multiclass variable
    # all_data_binary_param = param_binary_groups(data_sorted, 'FatherBirthplace','Israel')
    # continuous variable
    all_data_categorical_param = param_groups_median_threshold(data_sorted, 'trustDMrt')

    stats_offer_by_param(all_data_categorical_param, 'ultimatumOffer', 'trustDMrt')
    print('')

    # all_data_categorical_param = param_groups_median_threshold(data_sorted, 'ultimatumInstructionRT')
    stats_offer_by_param(all_data_categorical_param, 'trustOffer', 'ultimatumOffer')

    # continuous variables
    plot_keys_vs_param(data_sorted, 'ultimatumOffer', ['trustOffer'])

if __name__ == "__main__":
    main()