from pyspark import SparkContext
import time
import itertools as it
import os
import sys
import copy
import math
import csv

def process_data(x):
    x = x.replace('"', '').split(",")
    if x == header:
        return ["DATE-CUSTOMER_ID", "PRODUCT_ID"]
    else:
        return [x[trans] + "-" + x[cust], int(x[prod])]


def gen_tuple(freq_tuple, len_tuple):
    tuple_list = []
    for i in it.combinations(freq_tuple, 2):
        tuple_set = set(i[0]).union(set(i[1]))
        if len(tuple_set) == len_tuple + 1:
            tuple_list.append(tuple(sorted(tuple_set)))
    tuple_list = sorted(set(tuple_list))
    return tuple_list


def pcy(partition, support, size):
    partition = copy.deepcopy(list(partition))
    p_threshold = math.ceil(support * (len(partition) / size))
    bucket_num = 10
    tuple_list = []

    # pass 1: create list of frequent singles, hash pairs into buckets and find frequent buckets
    single_dict = {}
    hash_table = [0] * bucket_num
    for basket in partition:
        for i in basket:
            if i in single_dict.keys():
                single_dict[i] += 1
            else:
                single_dict[i] = 1
        for j in it.combinations(basket, 2):
            j = tuple(sorted(j))
            hash_key = abs(hash(j) % bucket_num)
            hash_table[hash_key] += 1
    # gen list of frequent singles
    freq_single = []
    for i in single_dict.items:
        if i[1] >= p_threshold:
            freq_single.append(i[0])
    freq_single = sorted(freq_single)

    # find frequent buckets
    bit_map = [True if i >= p_threshold else False for i in hash_table]
    tuple_list += freq_single

    # pass 2: generate candidates from frequents and check if they are frequent
    # for pairs, only reserve those in frequent buckets
    len_tuple = 2
    cand_tuple = [0]

    while len(cand_tuple) != 0:
        tuple_dict = {}
        for basket in partition:
            freq_list = [i for i in basket if i in freq_single]

            if len(freq_list) >= len_tuple == 2:
                for j in it.combinations(freq_list, len_tuple):
                    j = tuple(sorted(j))
                    hash_key = abs(hash(j) % bucket_num)
                    if j in tuple_dict.keys() and bit_map[hash_key]:
                        tuple_dict[j] += 1
                    elif j not in tuple_dict.keys() and bit_map[hash_key]:
                        tuple_dict[j] = 1

            elif len(freq_list) >= len_tuple > 2:
                for j in cand_tuple:
                    if j in tuple_dict.keys() and set(j).issubset(set(basket)):
                        tuple_dict[j] += 1
                    elif j not in tuple_dict.keys() and set(j).issubset(set(basket)):
                        tuple_dict[j] = 1

        freq_tuple = sorted(dict(filter(lambda x: x[1] >= p_threshold, tuple_dict.items())).keys())
        cand_tuple = gen_tuple(freq_tuple, len_tuple)

        len_tuple += 1
        tuple_list += freq_tuple

    yield tuple_list


def count_cand(partition, candidates):
    partition = copy.deepcopy(list(partition))
    cand_dict = {}
    for basket in partition:
        for i in candidates:
            if tuple(i) in cand_dict.keys() and set(i).issubset(set(basket)):
                cand_dict[tuple(i)] += 1
            elif tuple(i) not in cand_dict.keys() and set(i).issubset(set(basket)):
                cand_dict[tuple(i)] = 1
    cand_list = [[list(k), v] for k, v in cand_dict.items()]
    yield cand_list


def gen_dict(list):
    item_list = [list]
    item_dict = dict()
    for i in item_list:
        for j in i:
            item = sorted(set(j))
            item_dict.setdefault(len(item), []).append(item)
    item_dict = dict(sorted(item_dict.items()))
    return item_dict.items()


# start
start_time = time.time()

os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# input arguments

filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]
# filter_threshold = 20
# support = 50
# input_file = 'ta_feng_all_months_merged.csv'
# output_file = 'output2.txt'
process_file = "customer_product.csv"

sc = SparkContext("local[*]", "task2")
sc.setLogLevel("WARN")

# Data preprocessing
header = (sc.textFile(input_file).first()).replace('"', '').split(",")
trans = header.index("TRANSACTION_DT")
cust = header.index("CUSTOMER_ID")
prod = header.index("PRODUCT_ID")
processed_data = sc.textFile(input_file).map(lambda x: process_data(x)).collect()

# create csv file
with open(process_file, "w", encoding="utf-8", newline='' "") as csv_file:
    writer = csv.writer(csv_file)
    for x in processed_data:
        writer.writerow(x)
csv_file.close()


# read csv file and create rdd
rdd = sc.textFile(process_file).filter(lambda x: x != "DATE-CUSTOMER_ID,PRODUCT_ID").map(lambda x: x.split(","))\
    .groupByKey().map(lambda x: [x[0], list(set(x[1]))]).filter(lambda x: len(x[1]) > filter_threshold)

size = rdd.count()


# phase 1: find frequent itemsets in subsets
candidates = rdd.map(lambda x: x[1]).mapPartitions(lambda x: pcy(x, support, size)) \
    .flatMap(lambda x: x).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).map(lambda x: x[0]) \
    .map(lambda x: list(x) if type(x) is tuple else [x]).sortBy(lambda x: (len(x), x)).collect()
# print(candidates)


# phase 2: find frequent itemsets in the whole dataset
frequents = rdd.map(lambda x: x[1]).mapPartitions(lambda x: count_cand(x, candidates)).flatMap(lambda x: x)\
    .map(lambda x: (tuple(x[0]), x[1])).reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support)\
    .map(lambda x: list(x[0])).sortBy(lambda x: (len(x), x)).collect()
# print(frequents)

# create final output

# candidates
output_cand = ""
for k, v in gen_dict(candidates):
    v.sort()
    for i in v:
        if len(i) == 1:
            output_cand += "('" + str(i[0]) + "'),"
        else:
            output_cand += str(tuple(i)) + ","
    output_cand = output_cand.rstrip(",") + "\n\n"

# frequents
output_freq = ""
for k, v in gen_dict(frequents):
    v.sort()
    for i in v:
        if len(i) == 1:
            output_freq += "('" + i[0] + "'),"
        else:
            output_freq += str(tuple(sorted(i))) + ","
    output_freq = output_freq.rstrip(",") + "\n\n"

# final output
output = ""
output += "Candidates:\n" + output_cand + "Frequent Itemsets:\n" + output_freq.rstrip("\n")
with open(output_file, "w+") as output_2:
    output_2.writelines(output)


# duration
end_time = time.time()
print(f'Duration: {end_time - start_time}')
