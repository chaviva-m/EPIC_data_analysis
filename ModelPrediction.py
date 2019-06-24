from pickle import load, dump
from collections import OrderedDict, defaultdict
from operator import itemgetter
import numpy as np
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn import metrics
from statistics import mean
from scipy.stats import sem
from tabulate import tabulate
import matplotlib.pyplot as plt
import re
from statistics import median
from OfferStatistics import param_binary_groups


# to visualize decision tree copy content of .dot file here -
# http://www.webgraphviz.com/

DATA_DIR = 'data\\'
DICT_FILE = 'data_dict'

EMOTIONS = ['happy', 'sad', 'neutral']

EPSILON = 0.0000001


''' extract features '''

def add_feature(part_id, feature_dict, feature, feature_names, feature_name):
    if feature_name not in feature_names:
        feature_names.append(feature_name)
    feature_dict[part_id].append(feature)


def add_feature_all_participants(data, feature_dict, feature_names, key):
    feature_name = key
    for part_id, part_data in data.items():
        feature = part_data[key]
        add_feature(part_id, feature_dict, feature, feature_names, feature_name)


def recording_emotions(data, feature_dict, feature_names, recording_key):
    for part_id, part_data in data.items():
        emotions = part_data[recording_key]
        for emotion,strength in emotions.items():
            if emotion not in EMOTIONS:
                continue
            feature_name = '{}: {}'.format(recording_key, emotion)
            add_feature(part_id, feature_dict, strength, feature_names,
                        feature_name)


def recording_emotions_diff(data, feature_dict, feature_names, recording_key, recordingBL_key):
    for part_id, part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recordingBL_key]
        for emotion,strength in emotions.items():
            if emotion not in EMOTIONS:
                continue
            difference = strength - emotions_bl[emotion]
            feature_name = '{} difference: {}'.format(recording_key, emotion)
            add_feature(part_id, feature_dict, difference, feature_names,
                        feature_name)


def recording_emotions_ratio(data, feature_dict, feature_names, recording_key, recordingBL_key):
    for part_id, part_data in data.items():
        emotions = part_data[recording_key]
        emotions_bl = part_data[recordingBL_key]
        for emotion,strength in emotions.items():
            if emotion not in EMOTIONS:
                continue
            strength_bl = emotions_bl[emotion]
            if strength_bl == 0:
                strength_bl += EPSILON
            ratio = strength / strength_bl
            feature_name = '{} ratio: {}'.format(recording_key, emotion)
            add_feature(part_id, feature_dict, ratio, feature_names,
                        feature_name)


def self_report_emotions(data, feature_dict, feature_names):
    for part_id, part_data in data.items():
        emotions = part_data['selfReport']
        for emotion,strength in emotions.items():
            feature_name = 'self report: {}'.format(emotion)
            add_feature(part_id, feature_dict, strength, feature_names,
                        feature_name)


def self_report_compressed_emotions(data, feature_dict, feature_names):
    for part_id, part_data in data.items():
        emotions = part_data['selfReport']
        happy = (emotions['happiness'] + emotions['amusement'])
        add_feature(part_id, feature_dict, happy, feature_names,
                    'self report sum: happiness + amusement')
        sad = (emotions['sadness'] + emotions['grief'])
        add_feature(part_id, feature_dict, sad, feature_names,
                    'self report sum: sadness + grief')
        neutral = (emotions['apathy'] + emotions['calm'])
        add_feature(part_id, feature_dict, neutral, feature_names,
                    'self report sum: apathy + calm')


def exp_emotion(data, feature_dict, feature_names):
    feature_names.append('experiment emotion')
    for part_id, part_data in data.items():
        feature_dict[part_id].append(part_data['emotion'][1])


def extract_features(data):
    feature_dict = defaultdict(list)
    feature_names = list()

    '''extract features'''
    exp_emotion(data, feature_dict, feature_names)
    self_report_compressed_emotions(data, feature_dict, feature_names)
    self_report_emotions(data, feature_dict, feature_names)

    # video features
    recording_emotions(data, feature_dict, feature_names, 'videoFreq')
    recording_emotions(data, feature_dict, feature_names, 'videoFreqTopPercent')
    recording_emotions_diff(data, feature_dict, feature_names, 'videoFreq', 'videoBLFreq')
    recording_emotions_ratio(data, feature_dict, feature_names, 'videoFreq', 'videoBLFreq')
    recording_emotions_diff(data, feature_dict, feature_names, 'videoProbMean', 'videoBLProbMean')
    recording_emotions_ratio(data, feature_dict, feature_names, 'videoProbMean', 'videoBLProbMean')
    recording_emotions_diff(data, feature_dict, feature_names, 'videoThresholdFreq', 'videoBLThresholdFreq')
    recording_emotions_ratio(data, feature_dict, feature_names, 'videoThresholdFreq', 'videoBLThresholdFreq')
    recording_emotions_diff(data, feature_dict, feature_names, 'videoFreqSkipBeg', 'videoBLFreqSkipBeg')
    recording_emotions_ratio(data, feature_dict, feature_names, 'videoFreqSkipBeg', 'videoBLFreqSkipBeg')
    recording_emotions_diff(data, feature_dict, feature_names, 'videoFreqTopPercent', 'videoBLFreqTopPercent')
    recording_emotions_ratio(data, feature_dict, feature_names, 'videoFreqTopPercent', 'videoBLFreqTopPercent')

    # audio features
    recording_emotions(data, feature_dict, feature_names, 'audio')
    recording_emotions_diff(data, feature_dict, feature_names, 'audio', 'audioBL')
    recording_emotions_ratio(data, feature_dict, feature_names, 'audio', 'audioBL')

    # performance features
    add_feature_all_participants(data, feature_dict, feature_names, 'ultimatumDMrt')
    add_feature_all_participants(data, feature_dict, feature_names, 'ultimatumInstructionRT')
    add_feature_all_participants(data, feature_dict, feature_names, 'trustDMrt')
    add_feature_all_participants(data, feature_dict, feature_names, 'trustInstructionRT')
    add_feature_all_participants(data, feature_dict, feature_names,'ultimatumOffer')

    add_feature_all_participants(data, feature_dict, feature_names, 'gender')
    add_feature_all_participants(data, feature_dict, feature_names, 'yearOfBirth')
    add_feature_all_participants(data, feature_dict, feature_names, 'introversion')
    add_feature_all_participants(data, feature_dict, feature_names, 'extroversion')
    add_feature_all_participants(data, feature_dict, feature_names, 'altruism')

    # demographic features
    for param in ['Birthplace', 'FatherBirthplace', 'MotherBirthplace']:
        data_binary_param = param_binary_groups(data,param,'Israel')
        add_feature_all_participants(data_binary_param, feature_dict,
                                     feature_names, param)
    # if we want to add these features, need to first use param_binary_groups
    # add_feature_all_participants(data, feature_dict, feature_names, 'status')
    # add_feature_all_participants(data, feature_dict, feature_names, 'education')
    # add_feature_all_participants(data, feature_dict, feature_names, 'steadyIncome')
    # add_feature_all_participants(data, feature_dict, feature_names, 'religion')

    '''convert to list'''
    feature_dict_sorted = OrderedDict(sorted(feature_dict.items(), key=itemgetter(0)))
    feature_list = np.array([v for v in feature_dict_sorted.values()])
    return feature_list, feature_names


def filter_features(features, feature_names, features_to_choose):
    filtered_features = defaultdict(list)
    filtered_feature_names = []
    for feature_to_choose in features_to_choose:
        for i,name in enumerate(feature_names):
            if re.search(feature_to_choose, name):
                filtered_feature_names.append(name)
                for j,sample in enumerate(features):
                    filtered_features[j].append(features[j][i])
    feature_dict_sorted = OrderedDict(sorted(filtered_features.items(), key=itemgetter(0)))
    feature_list = np.array([v for v in feature_dict_sorted.values()])
    return feature_list, filtered_feature_names


def get_labels(data, key):
    labels = []
    for part_id,part_data in data.items():
        labels.append(part_data[key])
    return np.array(labels)


''' train models '''

def train_with_cross_validation(x, y, regressor_obj):
    R2 = []
    RMSE = []
    # note that RobustScaler is robust to outliers
    scaler = RobustScaler() # RobustScaler()  # StandardScaler() # MinMaxScaler()
    x_norm = scaler.fit_transform(x.astype(float))
    cv = LeaveOneOut()  # KFold(n_splits=3,shuffle=False)
    for train_index, test_index in cv.split(x_norm):
        [x_train, x_test, y_train, y_test] = x_norm[train_index], x_norm[test_index], \
                                             y[train_index], y[test_index]
        regressor_obj.fit(x_train, y_train)
        R2.append(regressor_obj.score(x_test,y_test))
        y_pred = regressor_obj.predict(x_test)
        RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    '''
    The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of
    squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
    ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and
    it can be negative (because the model can be arbitrarily worse).
    A constant model that always predicts the expected value of y, disregarding
    the input features, would get a R^2 score of 0.0.
    Note that R2 returns 0 if sample size is 1, like in our case of the leave one out
    '''
    return [np.mean(RMSE), sem(RMSE), np.mean(R2)]


def predict_mean_train_with_cross_validation(x, y):
    R2 = []
    RMSE = []
    cv = LeaveOneOut() # KFold(n_splits=3, shuffle=False)
    for train_index, test_index in cv.split(x):
        [x_train, x_test, y_train, y_test] = x[train_index], x[test_index], \
                                             y[train_index], y[test_index]
        mean_y_train = np.mean(y_train)
        y_pred = np.array([mean_y_train] * len(x_test))
        # R2.append(1-(sum(((y_test-y_pred)**2))/(sum((y_test-mean(y_test))**2))))
        RMSE.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    return [np.mean(RMSE), sem(RMSE), 0]  # np.mean(R2)]


def predict_mean(x, y):
    mean_y = np.mean(y)
    y_pred = np.array([mean_y] * len(x))
    R2 = (1-(sum(((y-y_pred)**2))/(sum((y-mean(y))**2))))
    RMSE = (np.sqrt(metrics.mean_squared_error(y, y_pred)))
    ste_RMSE = float('inf')     # one sample
    return [RMSE, ste_RMSE, R2]


def save_model_for_demo(data, label_key, model_name):
    [features, feature_names] = extract_features(data)
    labels = get_labels(data, label_key)
    features_to_choose = ['videoFreqTopPercent:']
    filtered_features, filtered_feature_names = filter_features(features,
                                                                feature_names,
                                                                features_to_choose)
    x = filtered_features
    y = labels
    # model = KNeighborsRegressor(n_neighbors=5)
    model = SVR(kernel='sigmoid', gamma=0.1)
    scaler = RobustScaler()
    x_norm = scaler.fit_transform(x.astype(float))
    model.fit(x_norm, y)
    dump(model, open(model_name, 'wb'))


''' compare models and features '''


def compare_models(data_sorted):

    [features, feature_names] = extract_features(data_sorted)
    ultimatum_labels = get_labels(data_sorted, 'ultimatumOffer')
    trust_labels = get_labels(data_sorted, 'trustOffer')
    feature_names_to_choose = ['audio diff']
    filtered_features, filtered_feature_names = filter_features(features, feature_names, feature_names_to_choose)

    features = filtered_features
    feature_names = filtered_feature_names

    print('features: {}'.format(feature_names), end='\n\n')

    # models
    svr = SVR(kernel='sigmoid', gamma=0.1)
    dtr = DecisionTreeRegressor(max_depth=3, random_state=1)
    rf = RandomForestRegressor(max_depth=2, random_state=1, n_estimators=100)   #1000
    lr = LinearRegression()
    knn = KNeighborsRegressor(n_neighbors=5)
    nn = MLPRegressor(hidden_layer_sizes=(features.shape[1],features.shape[1],), solver='lbfgs',
                      random_state=1, shuffle=False, alpha=0.000000001)

    # ultimatum
    ult_svr = train_with_cross_validation(features, ultimatum_labels, svr)
    ult_dtr = train_with_cross_validation(features, ultimatum_labels, dtr)
    export_graphviz(dtr, out_file='tree_ult.dot',
                    feature_names=feature_names)
    ult_rf = train_with_cross_validation(features, ultimatum_labels, rf)
    ult_lr = train_with_cross_validation(features, ultimatum_labels, lr)
    ult_knn = train_with_cross_validation(features, ultimatum_labels, knn)
    ult_nn = train_with_cross_validation(features, ultimatum_labels, nn)
    ult_apmt = predict_mean_train_with_cross_validation(features,
                                                        ultimatum_labels)
    ult_apm = predict_mean(features, ultimatum_labels)

    # trust
    trust_svr = train_with_cross_validation(features, trust_labels, svr)
    trust_dtr = train_with_cross_validation(features, trust_labels, dtr)
    export_graphviz(dtr, out_file='tree_trust.dot',
                    feature_names=feature_names)
    trust_rf = train_with_cross_validation(features, trust_labels, rf)
    trust_lr = train_with_cross_validation(features, trust_labels, lr)
    trust_knn = train_with_cross_validation(features, trust_labels, knn)
    trust_nn = train_with_cross_validation(features, trust_labels, nn)
    trust_apmt = predict_mean_train_with_cross_validation(features,
                                                          trust_labels)
    trust_apm = predict_mean(features, trust_labels)

    # print results in table
    t = tabulate([['svr'] + ult_svr + trust_svr,
                  ['decision tree'] + ult_dtr + trust_dtr,
                  ['random forest'] + ult_rf + trust_rf,
                  ['linear regression'] + ult_lr + trust_lr,
                  ['knn'] + ult_knn + trust_knn,
                  ['neural network'] + ult_nn + trust_nn,
                  ['predict mean of train'] + ult_apmt + trust_apmt,['predict mean'] + ult_apm + trust_apm],
                 headers=['model', 'ultimatum RMSE', 'ultimatum ste(RMSE)', 'ultimatum R2',
                          'trust RMSE', 'trust ste(RMSE)', 'trust R2'], tablefmt='orgtbl')

    print(t)


def compare_features(data_sorted):
    svr = SVR(kernel='sigmoid', gamma=0.1)
    [features, feature_names] = extract_features(data_sorted)
    ultimatum_labels = get_labels(data_sorted, 'ultimatumOffer')
    trust_labels = get_labels(data_sorted, 'trustOffer')

    table_rows = []
    all_features_to_choose = [['experiment emotion'],
                              ['self report:'],
                              ['videoFreq:'],
                              ['videoFreq diff'],['videoFreq ratio'],
                              ['videoThresholdFreq diff'],['videoThresholdFreq ratio'],
                              ['videoFreqSkipBeg diff'], ['videoFreqSkipBeg ratio'],
                              ['videoFreqTopPercent:'],
                              ['videoFreqTopPercent diff'], ['videoFreqTopPercent ratio'],
                              ['videoProbMean diff'], ['videoProbMean ratio'],
                              ['audio:'],
                              ['audio diff'],['audio ratio'],['audio diff','gender'],
                              ['ultimatumDMrt'],['ultimatumInstructionRT'],
                              ['trustDMrt'],['trustInstructionRT'],
                              ['altruism'], ['introversion'],
                              ['Birthplace'], ['FatherBirthplace'], ['MotherBirthplace'],
                              ['yearOfBirth', 'gender'],['gender']]
    for features_to_choose in all_features_to_choose:
        filtered_features, filtered_feature_names = filter_features(features, feature_names, features_to_choose)
        ult = train_with_cross_validation(filtered_features, ultimatum_labels, svr)
        trust = train_with_cross_validation(filtered_features, trust_labels, svr)
        table_rows += [[features_to_choose] + ult + trust]

    # mark best result and sort by ultimatum (1) or trust (4) or features (0)
    table_rows.sort(key=itemgetter(4))

    # print table
    t = tabulate(table_rows, headers = ['features', 'ultimatum mean RMSE', 'ultimatum SE RMSE',
                                        'ultimatum R2', 'trust mean RMSE', 'trust SE RMSE',
                                        'trust R2'], tablefmt = 'orgtbl')
    print(t)


def compare_features_with_dif_models(data_sorted, label_key, all_features_to_choose):
    [features, feature_names] = extract_features(data_sorted)
    labels = get_labels(data_sorted, label_key)

    features_models_RMSE = OrderedDict()

    for name,features_to_choose in all_features_to_choose.items():
        filtered_features, filtered_feature_names = filter_features(features, feature_names, features_to_choose)
        features_models_RMSE[name] = compare_models_for_features(filtered_features, labels)
    return features_models_RMSE


def compare_models_for_features(features, labels):

    # models
    svr = SVR(kernel='sigmoid', gamma=0.1)
    dtr = DecisionTreeRegressor(max_depth=3, random_state=1)
    rf = RandomForestRegressor(max_depth=2, random_state=1, n_estimators=100)
    lr = LinearRegression()
    knn = KNeighborsRegressor(n_neighbors=5)
    nn = MLPRegressor(
        hidden_layer_sizes=(features.shape[1], features.shape[1],),
        solver='lbfgs',
        random_state=1, shuffle=False, alpha=0.000000000001)
    #
    models = {'svr': svr, 'decision tree': dtr, 'random forest': rf,
             'linear regression': lr, 'knn': knn, 'neural netowrk': nn}
    # models = {'svr': svr, 'decision tree': dtr, 'random forest': rf,
    #          'knn': knn, 'neural netowrk': nn}
    models_RMSE = {}

    # train models
    for name,model in models.items():
        models_RMSE[name] = train_with_cross_validation(features, labels, model)[0:2]

    models_RMSE['predict mean'] = predict_mean_train_with_cross_validation(features,
                                                        labels)[0:2]
    # models_RMSE['predict mean of whole dataset'] = predict_mean(features, labels)[0:2]

    return models_RMSE


def bar_graph_features_models_RMSE(features_models_RMSE, graph_title):

    features_models_RMSE_sorted = features_models_RMSE

    width = 0.1
    N = len(features_models_RMSE_sorted.keys())   # number of models

    models_features_y = defaultdict(list)
    models_features_y_ste = defaultdict(list)
    for features, models in features_models_RMSE_sorted.items():
        for model,RMSE in models.items():
            models_features_y[model].append(RMSE[0])
            models_features_y_ste[model].append(RMSE[1])

    ind = np.arange(N)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    axs_models = []
    for i, ((model,y),y_ste) in enumerate(zip(models_features_y.items(), models_features_y_ste.values())):
        axs_models.append(ax.bar(ind + (i * width), y, width, yerr=y_ste))

    font_size = 14
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_ylabel('RMSE', fontsize=font_size)
    ax.set_xticks(ind + len(features_models_RMSE_sorted.keys())*width)
    ax.set_xticklabels( list(features_models_RMSE_sorted.keys()), fontsize=font_size)
    ax.legend( axs_models, list(features_models_RMSE_sorted.values())[0])
    ax.set_title(graph_title)


'''filter data'''


def binary_filter(data, param, value):
    data_filtered = OrderedDict()
    for j, d in data.items():
        if d[param] == value:
            data_filtered[j] = d
    return data_filtered


def filter_data(data, desired_participants):
    filtered_data = OrderedDict()
    for part_id, part_data in data.items():
        if part_id in desired_participants:
            filtered_data[part_id] = part_data
    return filtered_data


def low_pass_filter(data_sorted, param, divide_func):
    all_values = [part_data[param] for part_data in data_sorted.values()]
    threshold = divide_func(all_values)
    data_filtered = OrderedDict()
    for j, d in data_sorted.items():
        if d[param] < threshold:
            data_filtered[j] = d
    return data_filtered


def high_pass_filter(data_sorted, param, threshold_func):
    all_values = [part_data[param] for part_data in data_sorted.values()]
    threshold = threshold_func(all_values)
    data_filtered = OrderedDict()
    for j, d in data_sorted.items():
        if d[param] > threshold:
            data_filtered[j] = d
    return data_filtered


''' main '''


def main():
    with open(DATA_DIR + DICT_FILE, 'rb') as f:
        data = load(f)
    print('all: n={}'.format(len(data)))
    # data_filtered = low_pass_filter(data, 'altruism', mean)
    # data_filtered = high_pass_filter(data, 'introversion', mean)
    # data_filtered = low_pass_filter(data, 'extroversion', mean)
    # data_filtered = low_pass_filter(data, 'trustDMrt', median)
    # data_filtered = low_pass_filter(data, 'ultimatumDMrt', median)
    data_sorted = OrderedDict(sorted(data.items(), key=itemgetter(0)))
    print('after filter: n={}'.format(len(data_sorted)))

    # save_model_for_demo(data, 'ultimatumOffer', 'video_svr_model.sav')

    # # compare_models(data_sorted)

    # print('\n\n')

    compare_features(data_sorted)

    # ultimatum
    label_key = 'ultimatumOffer'
    graph_title = 'Ultimatum Offer Prediction'
    all_features_to_choose = OrderedDict([
        ('emotion\nself report', ['self report:']),
        ('emotion\nvideo', ['videoFreqTopPercent:']),
        ('emotion\naudio', ['audio diff']),
        ('gender', ['gender'])
    ])
    features_models_RMSE = compare_features_with_dif_models(data_sorted, label_key, all_features_to_choose)
    bar_graph_features_models_RMSE(features_models_RMSE, graph_title)

    # trust
    label_key = 'trustOffer'
    graph_title = 'Trust Offer Prediction'
    all_features_to_choose = OrderedDict([
        ('emotion\nself report', ['self report:']),
        ('emotion\nvideo', ['videoFreqTopPercent diff']),
        ('emotion\naudio', ['audio:']),
        ('gender', ['gender'])
    ])
    features_models_RMSE = compare_features_with_dif_models(data_sorted, label_key, all_features_to_choose)
    bar_graph_features_models_RMSE(features_models_RMSE, graph_title)
    plt.show()


if __name__ == "__main__":
    main()