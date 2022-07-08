from pyspark import SparkContext
import os
import sys
import json
import time
import numpy as np
import xgboost as xgb

# deal with NaN, infinite, null data, set default
def checkvalue(test_value):
    for i in range(len(test_value)):
        if np.isnan(test_value[i]):
            test_value[i] = 3  # assign default value
        elif test_value[i] == float("inf") or i == float("-inf"):
            test_value[i] = 3  # assign default value
        elif not test_value[i]:
            test_value[i] = 3

    return test_value


# time start
start = time.time()

# set environment
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# input arguments

# folder_path = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1"
# test_file = "yelp_test.csv"
# output_file = "test.csv"

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

sc = SparkContext("local[*]", "task2")
sc.setLogLevel("WARN")

# find train file, user file, business file under folder path
train_file = os.path.join(folder_path, "yelp_train.csv")
user_file = os.path.join(folder_path, "user.json")
busi_file = os.path.join(folder_path, "business.json")

# create train data rdd, remove the header
data_train = sc.textFile(train_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != "user_id")
# create test data rdd, remove the header
data_test = sc.textFile(test_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != "user_id")

# create business data rdd, use business id, starts, review count, location (lon+lat) as key features, return as dict
data_busi = sc.textFile(busi_file).map(lambda x: json.loads(x)) \
    .map(lambda x: (x["business_id"], [x["latitude"], x["longitude"], x["stars"], x["review_count"], ])).collectAsMap()

# create user data rdd, use user id, review count, and average stars as key features, return as dict
data_user = sc.textFile(user_file).map(lambda x: json.loads(x)) \
    .map(lambda x: (x["user_id"], [x["review_count"], x["average_stars"]])).collectAsMap()

# set x-axis and y-axis value for modelling
# should change the data as array for XGB model
x = data_train.map(lambda x: data_user[x[0]] + data_busi[x[1]]).map(lambda x: np.array(x)).collect()

# y is rating value
y = data_train.map(lambda x: float(x[2])).collect()


# create XGB model
model = xgb.XGBRegressor(max_depth=9, alpha=0, learning_rate=0.1, eval_metric="rmse", random_state=0)

# confirm the data is array
data_x = np.array(x)
data_y = np.array(y)

# fit the model
model.fit(data_x, data_y)


# check test data
# create list of user information and business information
user_busi = data_test.map(lambda x: (x[0], x[1])).collect()

# create test data rdd and change the data to array
x_test = data_test.map(lambda x: data_user[x[0]] + data_busi[x[1]]).map(lambda x: np.array(x)).collect()

# check predict y with real y in test data
data_x_t = np.array(x_test)
data_y_t = checkvalue(model.predict(data_x_t))
len_test = len(data_y_t)

# write output file
file = open(output_file, "w")
file.write("user_id, business_id, prediction\n")
for i in range(len_test):
    file.write(",".join(user_busi[i]) + "," + str(data_y_t[i]) + "\n")
file.close()

# time end
end = time.time()
print("Duration:", end - start)
