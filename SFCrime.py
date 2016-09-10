
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from copy import deepcopy


# Import Data
trainDF = pd.read_csv("sf_train.csv")
print("Finish loading.")
print(trainDF.head())
print(trainDF.describe())

# Clean up wrong X and Y values
xy_scalar = preprocessing.StandardScaler()
xy_scalar.fit(trainDF[["X", "Y"]])
trainDF[["X", "Y"]] = xy_scalar.transform(trainDF[["X", "Y"]])
trainDF = trainDF[abs(trainDF["Y"]) < 100]


# Create random sample from training data to plot
plot_data = trainDF.sample(frac=0.01, replace=True)
print(plot_data["X"].head())
plt.plot(plot_data["X"], plot_data["Y"], '.')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# ######################### Functions to Process Data ###########################

def parse_time(x):
    date_time = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    time = date_time.hour  # * 60 + date_time.minute
    day = date_time.day
    month = date_time.month
    year = date_time.year
    return time, day, month, year


def get_season(x):
    summer = 0
    fall = 0
    winter = 0
    spring = 0
    if x in [5, 6, 7]:
        summer = 1
    if x in [8, 9, 10]:
        fall = 1
    if x in [11, 0, 1]:
        winter = 1
    if x in [2, 3, 4]:
        spring = 1
    return summer, fall, winter, spring


def parse_data(df, logodds, logoddsPA):
    # Remove useless columns
    feature_list = df.columns.tolist()
    if "Descript" in feature_list:
        feature_list.remove("Descript")
    if "Resolution" in feature_list:
        feature_list.remove("Resolution")
    if "Category" in feature_list:
        feature_list.remove("Category")
    if "Id" in feature_list:
        feature_list.remove("Id")

    clean_data = df[feature_list]
    print("Creating address features")
    address_features = clean_data["Address"].apply(lambda x: logodds[x])
    address_features.columns = ["logodds" + str(x) for x in range(len(address_features.columns))]
    print("Parsing dates")
    clean_data["Time"], clean_data["Day"], clean_data["Month"], clean_data["Year"] \
        = zip(*clean_data["Dates"].apply(parse_time))

    # dummy_ranks_DAY = pd.get_dummies(cleanData['DayOfWeek'], prefix='DAY')
    # days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # cleanData["DayOfWeek"]=cleanData["DayOfWeek"].apply(lambda x: days.index(x)/float(len(days)))

    print("Creating one-hot variables")
    dummy_ranks_pd = pd.get_dummies(clean_data['PdDistrict'], prefix='PD')
    dummy_ranks_day = pd.get_dummies(clean_data["DayOfWeek"], prefix='DAY')
    dummy_ranks_h = pd.get_dummies(clean_data['Time'], prefix='Hr')
    dummy_ranks_m = pd.get_dummies(clean_data['Month'], prefix='M')
    dummy_ranks_date = pd.get_dummies(clean_data['Day'], prefix='D')
    clean_data["IsIntersection"] = clean_data["Address"].apply(lambda x: 1 if "/" in x else 0)
    clean_data["logoddsPA"] = clean_data["Address"].apply(lambda x: logoddsPA[x])

    print("dropping processed columns")
    clean_data = clean_data.drop("PdDistrict", axis=1)
    clean_data = clean_data.drop("DayOfWeek", axis=1)
    clean_data = clean_data.drop("Address", axis=1)
    clean_data = clean_data.drop("Dates", axis=1)
    # clean_data = clean_data.drop("Time", axis=1)
    clean_data = clean_data.drop("Day", axis=1)
    clean_data = clean_data.drop("Month", axis=1)
    feature_list = clean_data.columns.tolist()

    print("joining one-hot features")
    # features = cleanData[feature_list].join(dummy_ranks_PD.ix[:,:]).join(dummy_ranks_DAY.ix[:,:])\
    # .join(address_features.ix[:,:])
    features = clean_data[feature_list].join(dummy_ranks_h.ix[:, :]).join(dummy_ranks_m.ix[:, :])\
        .join(dummy_ranks_date.ix[:, :]).join(dummy_ranks_pd.ix[:, :]).join(dummy_ranks_day.ix[:, :])\
        .join(address_features.ix[:, :])

    print("creating new features")
    features["IsDup"] = pd.Series(features.duplicated() | features.duplicated(take_last=True)).apply(int)
    features["DayTime"] = features["Time"].apply(lambda x: 1 if (x == 0 or 8 <= x <= 23) else 0)
    features = features.drop("Time", axis=1)
    # features["Summer"], features["Fall"], features["Winter"], features["Spring"]
    #     = zip(*features["Month"].apply(get_season))
    # features = features.drop("Month", axis=1)

    if "Category" in df.columns:
        labels = df["Category"].astype('category')
    else:
        labels = None
    return features, labels


# ############################ Process Data ##################################

addresses = sorted(trainDF["Address"].unique())
categories = sorted(trainDF["Category"].unique())
cat_counts = trainDF.groupby(["Category"]).size()
addr_cat_counts = trainDF.groupby(["Address", "Category"]).size()
addr_counts = trainDF.groupby(["Address"]).size()
logodds = {}
logoddsPA = {}
MIN_CAT_COUNTS = 2

# Transform address to count-based feature
default_logodds = np.log(cat_counts / len(trainDF)) - np.log(1.0 - cat_counts / float(len(trainDF)))
for addr in addresses:
    PA = addr_counts[addr] / float(len(trainDF))
    logoddsPA[addr] = np.log(PA) - np.log(1. - PA)
    logodds[addr] = deepcopy(default_logodds)

    for cat in addr_cat_counts[addr].keys():
        if (addr_cat_counts[addr][cat] > MIN_CAT_COUNTS) and addr_cat_counts[addr][cat] < addr_counts[addr]:
            PA = addr_cat_counts[addr][cat] / float(addr_counts[addr])
            logodds[addr][categories.index(cat)] = np.log(PA) - np.log(1.0 - PA)

    logodds[addr] = pd.Series(logodds[addr])
    logodds[addr].index = range(len(categories))


features, labels = parse_data(trainDF, logodds, logoddsPA)

print(features.columns.tolist())
print(len(features.columns))


col_list = features.columns.tolist()
scalar = preprocessing.StandardScaler()
scalar.fit(features)
features[col_list] = scalar.transform(features)


sss = StratifiedShuffleSplit(labels, train_size=0.5)
for train_index, test_index in sss:
    features_train = features.iloc[train_index]
    features_test = features.iloc[test_index]
    labels_train = labels[train_index]
    labels_test = labels[test_index]


def build_and_fit_model(x_train, y_train, x_test=None, y_test=None, hn=32, dp=0.5, layers=1, epochs=1, batches=64, verbose=0):
    input_dim = x_train.shape[1]
    output_dim = len(labels_train.unique())
    Y_train = np_utils.to_categorical(y_train.cat.rename_categories(range(len(y_train.unique()))))
    # print(output_dim)
    model = Sequential()
    model.add(Dense(hn, input_shape=(input_dim,)))
    model.add(PReLU())
    model.add(Dropout(dp))

    for i in range(layers):
        model.add(Dense(hn))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dp))

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    if x_test is not None:
        Y_test = np_utils.to_categorical(y_test.cat.rename_categories(range(len(y_test.unique()))))
        fitting = model.fit(x_train, Y_train, nb_epoch=epochs, batch_size=batches, verbose=verbose, validation_data=(x_test, Y_test))
        test_score = log_loss(y_test, model.predict_proba(x_test, verbose=0))
    else:
        model.fit(x_train, Y_train, nb_epoch=epochs, batch_size=batches, verbose=verbose)
        fitting = 0
        test_score = 0

    return test_score, fitting, model


# ########################### Build and Fit Model #############################

N_EPOCHS = 20
N_HN = 256
N_LAYERS = 1
DP = 0.5

# Neural Net
score, fitting, model = build_and_fit_model(features_train.as_matrix(), labels_train, X_test=features_test.as_matrix(),
                                            y_test=labels_test, hn=N_HN, layers=N_LAYERS, epochs=N_EPOCHS, verbose=2,
                                            dp=DP)
# Logistic Regression
# model = LogisticRegression()
# model.fit(features_train, labels_train)


print("all", log_loss(labels, model.predict_proba(features.as_matrix())))
print("train", log_loss(labels_train, model.predict_proba(features_train.as_matrix())))
print("test", log_loss(labels_test, model.predict_proba(features_test.as_matrix())))


# ############################## Train the Final Model #################################


score, fitting, model = build_and_fit_model(features.as_matrix(), labels, hn=N_HN, layers=N_LAYERS, epochs=N_EPOCHS,
                                            verbose=2, dp=DP)
# model.fit(features, labels)


print("all", log_loss(labels, model.predict_proba(features.as_matrix())))
print("train", log_loss(labels_train, model.predict_proba(features_train.as_matrix())))
print("test", log_loss(labels_test, model.predict_proba(features_test.as_matrix())))


# ################################ Load Testing Data ###################################

testDF = pd.read_csv("sf_test.csv")
testDF[["X", "Y"]] = xy_scalar.transform(testDF[["X", "Y"]])
testDF["X"] = testDF["X"].apply(lambda x: 0 if abs(x) > 5 else x)
testDF["Y"] = testDF["Y"].apply(lambda y: 0 if abs(y) > 5 else y)


new_addresses = sorted(testDF["Address"].unique())
new_a_counts = testDF.groupby("Address").size()
only_new = set(new_addresses + addresses) - set(addresses)
only_old = set(new_addresses + addresses) - set(new_addresses)
in_both = set(new_addresses).intersection(addresses)

for addr in only_new:
    PA = new_a_counts[addr] / float(len(testDF) + len(trainDF))
    logoddsPA[addr] = np.log(PA) - np.log(1. - PA)
    logodds[addr] = deepcopy(default_logodds)
    logodds[addr].index = range(len(categories))

for addr in in_both:
    PA = (addr_counts[addr] + new_a_counts[addr]) / float(len(testDF) + len(trainDF))
    logoddsPA[addr] = np.log(PA) - np.log(1. - PA)


features_sub, _ = parse_data(testDF, logodds, logoddsPA)

col_list = features_sub.columns.tolist()
print(col_list)


features_sub[col_list] = scalar.transform(features_sub[col_list])

# ############################# Make Prediction ###############################

predDF = pd.DataFrame(model.predict_proba(features_sub.as_matrix()), columns=sorted(labels.unique()))


print(predDF.head())
predDF.to_csv("submission.csv", index_label="Id", na_rep="0")
