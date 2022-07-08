from pyspark import SparkContext
import time
import itertools as it
import os
import sys
import random

# time start
start = time.time()

# set environment
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# input arguments

# input_file = "yelp_train_test.csv"
# output_file = "test.csv"
input_file = sys.argv[1]
output_file = sys.argv[2]

sc = SparkContext("local[*]", "task1")
sc.setLogLevel("WARN")

# read the data of input file
data_raw = sc.textFile(input_file).map(lambda x: x.split(','))

# remove the header line
header = data_raw.first()
data = data_raw.filter(lambda x: x != header).cache()

# find distinct users
user = data.map(lambda x: x[0]).distinct().collect()

# assign index for each user
user_index = dict()
for i, j in enumerate(user):
    user_index[j] = i

# band, row, hash number
band = 100
row = 2
hash_num = row * band

# create hash function list for user index
user_index_h = dict()
hash_t = 0
while hash_t < hash_num:
    p = 57469
    m = 46853
    a = random.randint(101, 101010)
    b = random.randint(102, 102020)

    for i in range(len(user)):
        if i in user_index_h.keys():
            user_index_h[i].append(((a * i + b) % p) % m)
        else:
            user_index_h[i] = [((a * i + b) % p) % m]
    hash_t += 1

# create signature
sig = data.map(lambda x: (str(x[0]), str(x[1]))).map(lambda x: (x[1], user_index[x[0]])).groupByKey()\
    .map(lambda x: (x[0], list(set(x[1])))).map(lambda x: (x[0], [user_index_h[i] for i in x[1]]))\
    .map(lambda x: (x[0], [min(i) for i in zip(*x[1])])).collect()


# find candidates that share at least one same bucket
# create a set for candidates
cand = set()

# create bucket dict in each band
for i in range(band):
    buck = dict()
# find signature of the bucket and then hash to dict
    for j in sig:
        buck_sig = str()
        for k in range(row):
            buck_sig = buck_sig + str(j[1][row * i + k])
        if hash(buck_sig) in buck.keys():
            buck[hash(buck_sig)].add(j[0])
        else:
            buck[hash(buck_sig)] = set()
            buck[hash(buck_sig)].add(j[0])
# find candidates in the bucket
    for j in buck.values():
        if len(j) >= 2:
            for k in it.combinations(j, 2):
                cand.add(tuple(sorted(k)))

# calculate the similarity
# create a business-user dict with user index set
data_index = data.map(lambda x: (x[1], user_index[x[0]])).groupByKey().map(lambda x: (x[0], set(x[1]))).collectAsMap()

# create list for the result
result = []

# find candidates with similarity >= 0.5
# calculate the intersection and union item number
for i in cand:
    len_inter = len(data_index[i[0]].intersection(data_index[i[1]]))
    len_union = len(data_index[i[0]].union(data_index[i[1]]))
# calculate similarity
    sim = float(len_inter/len_union)
    if sim >= 0.5:
        result.append((i[0], i[1], str(sim)))

# create output file
file = open(output_file, "w")
file.write("business_id_1, business_id_2, similarity\n")
# Sort the results
result.sort()
# write to file
for i in result:
    file.write(",".join(i) + "\n")
file.close()

# time end
end = time.time()
print("Duration:", end - start)




