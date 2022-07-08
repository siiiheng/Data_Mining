# Method Description:
# I used XGBRegressor to create model-based recommendation system. Compared with assignment 3, I added more relevant features
# such as tip and check in data. Also, I used some statistical values to do the feature engineering, like mean, std, min, and max.
# I didn't use the hybrid recommendation model as the assignment 3.2.3 because it didn't improve the performance.

# Error Distribution:
# >=0 and <1: 102242
# >=1 and <2: 32846
# >=2 and <3: 6170
# >=3 and <4: 785
# >=4: 1

# RMSE:
# 0.97806

# Execution Time:
# 500s


from pyspark import SparkContext
import time
import pandas as pd
from xgboost import XGBRegressor
import json
import sys
import os
import numpy as np
import math

# use the four stats numbers to do feature engineering
def addstats(uid, bid):
    first = {"user_id": uid, "business_id": bid}

    # get user list
    user = train_user.get(uid, [])
    user = [i for i in user if i[0] != bid]

    # get business list
    busi = train_busi.get(bid, [])
    busi = [i for i in busi if i[0] != uid]

    # use stats to do user feature engineering
    user_feature = dict()
    if len(user) > 0:
        array = pd.Series([float(i[1]) for i in user])
        user_feature["u_avg"] = array.mean()
        user_feature["u_std"] = array.std()
        user_feature["u_max"] = array.max()
        user_feature["u_min"] = array.min()
    else:
        user_feature = default_user

    # use stats to do business feature engineering
    busi_feature = dict()
    if len(busi) > 0:
        array = pd.Series([float(i[1]) for i in busi])
        busi_feature["b_avg"] = array.mean()
        busi_feature["b_std"] = array.std()
        busi_feature["b_max"] = array.max()
        busi_feature["b_min"] = array.min()
    else:
        busi_feature = default_busi

    # combine user features and business features
    combine_feature = dict(user_feature, **busi_feature)

    return dict(first, **combine_feature)

# use the four stats numbers to do feature engineering
def calstats(uid, bid, score, user_list, busi_list):
    # first two columns
    first = {"user_id": uid, "business_id": bid,'y': score}

    # calculates the stats for business
    business = busi_list.remove(score)
    busi_stats = dict()
    if business and len(business) > 0:
        array = pd.Series(business)
        busi_stats["b_avg"] = array.mean()
        busi_stats["b_std"] = array.std()
        busi_stats["b_max"] = array.max()
        busi_stats["b_min"] = array.min()
    else:
        busi_stats = default_busi

    # calculates the stats for users
    try:
        user = user_list.remove(score)
    except:
        print(user_list)
    user_stats = dict()
    if user and len(user) > 0:
        array = pd.Series(user)
        user_stats["u_avg"] = array.mean()
        user_stats["u_std"] = array.std()
        user_stats["u_max"] = array.max()
        user_stats["u_min"] = array.min()
    else:
        user_stats = default_user

    # combine all the dicts
    combine_stats = dict(first, **user_stats, **busi_stats)
    return combine_stats


# use default value to replace missing value
def missingvalue(df):
    df["b_avg"].fillna(default_busi["b_avg"], inplace=True)
    df["b_std"].fillna(default_busi["b_std"], inplace=True)
    df["b_max"].fillna(default_busi["b_max"], inplace=True)
    df["b_min"].fillna(default_busi["b_min"], inplace=True)
    df.fillna(0, inplace=True)

# calculate the duration of registration
def timediff(df, column):
    year = 2022 - 1 - df[column].transform(lambda x: int(x[:4]))
    month = year * 12 + df[column].transform(lambda x: int(x[5:7]))
    return month

# time start
start = time.time()

# set environment
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

sc = SparkContext("local[*]", "task")
sc.setLogLevel("WARN")

# input argument
folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

# find train file, user file, business file, tip file, checkin file under folder path
train_file = os.path.join(folder_path, "yelp_train.csv")
user_file = os.path.join(folder_path, "user.json")
busi_file = os.path.join(folder_path, "business.json")
tip_file = os.path.join(folder_path, "tip.json")
checkin_file = os.path.join(folder_path, "checkin.json")

# read train data and test data
train = sc.textFile(train_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != "user_id").cache()
test = sc.textFile(test_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != "user_id").cache()

# create full set of all users in train data and test data
user_train_list = train.map(lambda x: x[0]).distinct().collect()
user_test_list = test.map(lambda x: x[0]).distinct().collect()
# combine both user list to find all users set
user_total = set(user_train_list + user_test_list)

# create dicts for user: busi, score and busi: user, score
train_user = train.map(lambda x: (x[0], (x[1], x[2]))).map(lambda x: (x[0], x[1])).groupByKey().mapValues(lambda x: list(x)).collectAsMap()
train_busi = train.map(lambda x: (x[1], (x[0], x[2]))).map(lambda x: (x[0], x[1])).groupByKey().mapValues(lambda x: list(x)).collectAsMap()

# group the scores by user and business respectively of the train data
user_score = train.map(lambda x: (x[0], float(x[2]))).groupByKey()
busi_score = train.map(lambda x: (x[1], float(x[2]))).groupByKey()

# merge the user_score to train data
usermerge = train.map(lambda x: (x[0], (x[1], float(x[2])))).join(user_score).map(lambda x:(x[1][0][0],(x[0],x[1][0][1],x[1][1]))).cache()
# then merge the business_score to the previous rdd
busimerge = usermerge.join(busi_score).map(lambda x: (x[1][0][0], x[0], x[1][0][1], x[1][0][2], x[1][1])).cache()

# find key features from each file
# user key features
user_kf = sc.textFile(user_file).map(lambda x: json.loads(x)).filter(lambda x: x["user_id"] in user_total).collect()
# deal with none data in friends and elite columns, then delete name column
for i in user_kf:
    if i["friends"] == "None":
        i["friends"] = 0
    else:
        i["friends"] = len(i["friends"].split(","))
    if i["elite"] == "None":
        i["elite"] = 0
    else:
        i["elite"] = len(i["elite"].split(","))
    del i["name"]

# business features, only use stars, review_count, latitude, longitude, is_open
busi_kf = sc.textFile(busi_file).map(lambda x: json.loads(x))\
    .map(lambda x: (x["business_id"], x["stars"], x["review_count"], x["latitude"], x["longitude"], x["is_open"]))\
    .map(lambda x:{"business_id": x[0], "b_stars": x[1], "b_review_count": x[2], "latitude": x[3], "longitude": x[4], "is_open": x[5]})\
    .collect()

# tip features, calculate the total likes and how many times
tip = sc.textFile(tip_file).map(lambda x: json.loads(x)).map(lambda x: (x["business_id"], (x["likes"], 1)))\
    .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).map(lambda x: {"business_id": x[0],
                                                                         "likes_total": x[1][0],
                                                                         "likes_times": x[1][1]}).collect()

# checkin features, calculate how many people and how many days
checkin = sc.textFile(checkin_file).map(lambda x: json.loads(x))\
    .map(lambda x: (x["business_id"], len(list(x["time"].values())), sum(list(x["time"].values()))))\
    .map(lambda x: {"business_id": x[0], "days": x[1], "clients": x[2]}).collect()

# combine the features, user feature (to dataframe) separately
feature_list = [busi_kf, tip, checkin]
user_kf_df = pd.DataFrame(user_kf)

# set default value for stats terms
default_user = {"u_avg": 3.75, "u_std": 1, "u_max": 5, "u_min": 1}
default_busi = {"b_avg": 3.75, "b_std": 1, "b_max": 5, "b_min": 1}

# partition the merged rdd and calculate the statistical data, change it to dataframe
train_merge = pd.DataFrame(busimerge.repartition(50).map(lambda x: calstats(x[0], x[1], x[2], list(x[3]), list(x[4]))).collect())

# merge the feature dataframe to merged train data, then merge the user features to the data as well
for i in feature_list:
    feature_df = pd.DataFrame(i)
    train_merge = pd.merge(train_merge, feature_df, on="business_id", how="left")
train_data = pd.merge(train_merge, user_kf_df, on="user_id", how="left")

# deal with the missing data
missingvalue(train_data)

# add a column that shows how long the user uses yelp, divide the datestamp to year and month, then calculate the
# month duration
train_data["duration"] = timediff(train_data, "yelping_since")

# partition the test data and add the feature columns, change it to dataframe
# merge the feature dataframe to test data, then merge the user features to the data as well
test_merge = pd.DataFrame(test.repartition(20).map(lambda x: addstats(x[0], x[1])).collect())
for i in feature_list:
    feature_df = pd.DataFrame(i)
    test_merge = pd.merge(test_merge, feature_df, on="business_id", how="left")
test_data = pd.merge(test_merge, user_kf_df, on="user_id", how="left")

# deal with the missing data
missingvalue(test_data)

# add a column that shows how long the user uses yelp, divide the datestamp to year and month, then calculate the
# month duration
test_data["duration"] = timediff(test_data, "yelping_since")

# find valid feature columns
train_columns = train_data.columns
feature_columns = train_columns.difference(["y", "business_id", "user_id", "yelping_since"])

# prepare the train data to fit the model
X_train = train_data[feature_columns]
y_train = train_data["y"].astype("float")
model = XGBRegressor(colsample_bytree=0.5, max_depth=8, n_estimators=400, min_child_weight=4, reg_alpha=20,
                     subsample=0.5, gamma=0, reg_lambda=20, learning_rate=0.07)
model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_train, y_train)], verbose=50)

# model prediction
prediction = pd.DataFrame(model.predict(test_data[feature_columns]))
# first two columns
first = test_data[["user_id", "business_id"]]
# combine these two part in the result
result = pd.concat([first, prediction], axis=1)
result.columns = ["user_id", "business_id", "prediction"]
# output the result
result.to_csv(output_file, index=False)

# end time
end = time.time()
print(f"Execution time: {end-start}")

# check with validation data
val = pd.read_csv(test_file)
# merge the prediction value to the validation data
val_predict = val.merge(result, on=["user_id", "business_id"])
# column start is the ground value
ground = np.array(val_predict["stars"])
pred = np.array(val_predict["prediction"])

# calculate rmse between the groud value and prediction value
rmse = math.sqrt(sum((ground - pred) ** 2) / len(ground))
# find error distribution
# set categorical conditions
conditions = [
    (abs(val_predict["prediction"] - val_predict["stars"]) < 1),
    (abs(val_predict["prediction"] - val_predict["stars"]) >= 1) & (abs(val_predict["prediction"] - val_predict["stars"]) < 2),
    (abs(val_predict["prediction"] - val_predict["stars"]) >= 2) & (abs(val_predict["prediction"] - val_predict["stars"]) < 3),
    (abs(val_predict["prediction"] - val_predict["stars"]) >= 3) & (abs(val_predict["prediction"] - val_predict["stars"]) < 4),
    (abs(val_predict["prediction"] - val_predict["stars"]) >= 4)
    ]
# set categorical tags
values = [">=0 and <1", ">=1 and <2", ">=2 and <3", ">=3 and <4", ">=4"]

# fill in the dataframe and group by the categorical tags
val["error_distribution"] = np.select(conditions, values)
error_dis = val.groupby("error_distribution").size()

print(f"RMSE: {rmse}")
print(error_dis)

