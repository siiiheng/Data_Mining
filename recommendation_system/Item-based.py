from pyspark import SparkContext
import time
import os
import sys
import math


# create index, value dict for list
def listtodict(x):
    new_dict = dict()
    for i, j in enumerate(x):
        new_dict[j] = i
    return new_dict


# change key-value pair to value-key
def dict_reverse(x):
    new_dict = dict()
    for i, j in x.items():
        new_dict[j] = i
    return new_dict


# change a tuple list to dict
def tupletodict(x):
    new_dict = dict()
    for (i, j) in x:
        new_dict[i] = j
    return new_dict


# calculate pearson similarity
def pearson(x, y, d):
    x_rating = d[x]
    y_rating = d[y]

    # denominator is not 0
    if not x_rating or not y_rating:
        return 0

    # change tuple to key, value pair and find common rating
    x_kv = tupletodict(x_rating)
    y_kv = tupletodict(y_rating)
    common = set(x_kv.keys()).intersection(set(y_kv.keys()))

    # calculate average
    x_avg = sum(x_kv.values())/len(x_kv)
    y_avg = sum(y_kv.values())/len(y_kv)

    # remove those with fewer common items to speed up
    if len(common) < 70:
        return 0

    # calculate sum(x-xbar)*(y-ybar)
    sum_up = 0
    sum_x2 = 0
    sum_y2 = 0
    for i in common:
        sum_up += (x_kv[i] - x_avg) * (y_kv[i] - y_avg)
        sum_x2 += (x_kv[i] - x_avg) ** 2
        sum_y2 += (y_kv[i] - y_avg) ** 2

    # denominator is not 0
    if sum_x2 == 0 or sum_y2 == 0:
        return 0

    # calculate similarity
    sim = sum_up/(math.sqrt(sum_x2) * math.sqrt(sum_y2))

    return sim


# calculate the ratings with pearson
def cal_rate(corr, user, n):
    # set default rating
    rate = 3
    # if no pearson correlation, use default rating
    if not corr:
        return rate

    # if no positive pearson, use default rating
    corr = filter(lambda x: x[0] > 0, corr)
    if not corr:
        return rate

    # find candidates
    # sort pearson in descending
    corr = sorted(corr, key=lambda x: x[0], reverse=True)
    # candidates no more than n
    upper = min(n, len(corr))
    # find candidates
    candi = corr[0: upper]

    # calculate denominator
    sum_down = sum([abs(i) for (i, j) in candi])

    # if denominator is 0
    if sum_down == 0:
        # if no other user ratings, return default rating
        if not user:
            return rate
        # else return the average of ratings
        else:
            rate = sum([j for (i, j) in user])/len(user)
            return rate
    # if denominator is not 0, return weighted rating
    else:
        sum_up = sum([j * (i) for (i, j) in candi])
        rate = sum_up/sum_down
        return rate


# time start
start = time.time()

# set environment
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# input arguments

# train_file = "yelp_train.csv"
# val_file = "yelp_val.csv"
# output_file = "test.csv"

train_file = sys.argv[1]
val_file = sys.argv[2]
output_file = sys.argv[3]

sc = SparkContext("local[*]", "task2")
sc.setLogLevel("WARN")

# read file, create rdd for train data and val data
data_train = sc.textFile(train_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != "user_id")\
    .map(lambda x: (str(x[1]), str(x[0]), float(x[2]))).cache()  # confirm the data format
data_val = sc.textFile(val_file).map(lambda x: x.split(",")).filter(lambda x: x[0] != "user_id")\
    .map(lambda x: (str(x[1]), str(x[0]))).cache()

# find distinct businesses for train data and val data
busi_train = data_train.map(lambda x: x[0]).distinct().collect()
busi_val = data_val.map(lambda x: x[0]).distinct().collect()
# combine both business list, use set to find all distinct businesses
busi_total = sorted(set(busi_val + busi_train))
# assign index to businesses
busi_index = listtodict(busi_total)


# find distinct users for train data and val data
user_train = data_train.map(lambda x: x[1]).distinct().collect()
user_val = data_val.map(lambda x: x[1]).distinct().collect()
# combine both user list to find all users
user_total = sorted(set(user_val + user_train))
# assign index to users
user_index = listtodict(user_total)

# create business data dict
# create business train data with business id, tuple of user id and rating and return as key-value dict
data_busi = data_train.map(lambda x: (busi_index[x[0]], (user_index[x[1]], x[2]))).groupByKey()\
    .map(lambda x: (x[0], list(x[1]))).collectAsMap()


# create user data dict
# create user train data with user id, tuple of business id and rating and return as key-value dict
data_user = data_train.map(lambda x: (user_index[x[1]], (busi_index[x[0]], x[2]))).groupByKey()\
    .map(lambda x: (x[0], list(x[1]))).collectAsMap()

# set default list for missing keys
for i in busi_index.values():
    if i not in data_busi.keys():
        data_busi[i] = []
for j in user_index.values():
    if j not in data_user.keys():
        data_user[j] = []


# calculate pearson similarity for val data
val_pear = data_val.map(lambda x: (user_index[x[1]], busi_index[x[0]]))\
    .map(lambda x: (x[0], x[1], [(pearson(i, x[1], data_busi), j) for (i, j) in data_user[x[0]]]))

# calculate rating for val data
# find user with index id
user_finder = dict_reverse(user_index)
# find business with index id
busi_finder = dict_reverse(busi_index)
# calculate the rating
val_rate = val_pear.map(lambda x: (user_finder[x[0]], busi_finder[x[1]], cal_rate(x[2], data_user[x[0]], 2))).collect()


# write output file
file = open(output_file, "w")
file.write("user_id, business_id, prediction\n")
for i in val_rate:
    file.write(",".join([str(j) for j in i]) + "\n")
file.close()

# time end
end = time.time()
print("Duration:", end - start)
