from pyspark import SparkContext
import time
import os
import sys
import itertools as it
import graphframes
from pyspark.sql import SparkSession

# time start
start = time.time()

# set environment
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# input arguments

# thre = 7
# input_file = 'ub_sample_data.csv'
# output_file = 'test.txt'

thre = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]

sc = SparkContext("local[*]", "task1")
sc.setLogLevel("WARN")

# read file, create rdd
data = sc.textFile(input_file).map(lambda x: x.split(",")).cache()
# find header
header = data.first()
# remove header line from data
data_noh = data.filter(lambda x: x != header).map(lambda x: (x[0], x[1])).cache()

# get user-business data dictionary
user_busi = data_noh.groupByKey().collectAsMap()
# find distinct users
user_dis = data_noh.map(lambda x: x[0]).distinct().collect()

# filter candidate pairs no less than the threshold
cand = set()
for i in it.combinations(user_dis, 2):
    busi_union = set(user_busi[i[0]]).intersection(set(user_busi[i[1]]))
    if len(busi_union) >= thre:
        cand.add((i[0], i[1]))

# create a set for all users in the candidates and save as list
cand_user = set()
for i in cand:
    cand_user.add(i[0])
    cand_user.add(i[1])
cand_user_list = [i for i in cand_user]

# set sql sparksession
sqlsc = SparkSession.builder.getOrCreate()

# set schema for vertex and edge
vs = "id"
es = ["src", "dst"]

# create dataframe for the vertex and edge
vertex = sqlsc.createDataFrame(cand_user_list, "string").toDF(vs)
edge = sqlsc.createDataFrame(list(cand), es)

# create graphframe
graph = graphframes.GraphFrame(vertex, edge)
# use label propagation
res = graph.labelPropagation(maxIter=5)

# create communities output and do sorting
output = res.rdd.map(tuple).map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: sorted(x[1])).sortBy(lambda x: (len(x), x)).collect()

# write the output of communities to file
file = open(output_file, "w")
for i in output:
    file.write(",".join(["'" + j + "'" for j in i]))
    file.write("\n")

# end time
end = time.time()
print("Execution time:", end - start)
