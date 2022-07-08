from pyspark import SparkContext
import os
import json
import sys
import time

# set python environment
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# set Vocareum python environment
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

# set input and output path and number of partition
input_path = sys.argv[1]
output_path = sys.argv[2]
n_partition = int(sys.argv[3])

# set spark context
sc = SparkContext("local[*]", "task2")
sc.setLogLevel("WARN")

# create dict for final output
output = dict()

# results of default partition
output["default"] = dict()

# set start time
start_time = time.time()

# read file and create rdd file, saving to cache
review_rdd_de = sc.textFile(input_path).map(lambda x: json.loads(x)).cache()

# number of partitions
output["default"]["n_partition"] = review_rdd_de.getNumPartitions()
# number of items
output["default"]["n_items"] = review_rdd_de.glom().map(lambda x: len(x)).collect()
# task1 F
top10_businesses = review_rdd_de.map(lambda x: (x["business_id"], 1)).reduceByKey(lambda x, y: x+y).\
                                sortBy(lambda x: (-x[1], x[0])).take(10)
# calculate execution time
end_time = time.time()
output["default"]["exe_time"] = end_time - start_time


# results of customized partition
output["customized"] = dict()

# set start time
start_time = time.time()

# read file and create rdd file, then do partition by, saving to cache
review_rdd_cus = sc.textFile(input_path).map(lambda x: json.loads(x))\
    .map(lambda x: (x["business_id"], 1)).partitionBy(n_partition, lambda x: ord(x[0:1])).cache()

# task1 F
top10_businesses = review_rdd_cus.reduceByKey(lambda x, y: x+y).sortBy(lambda x: (-x[1], x[0])).take(10)

# number of partitions
output["customized"]["n_partition"] = review_rdd_cus.getNumPartitions()
# number of items
output["customized"]["n_items"] = review_rdd_cus.glom().map(lambda x: len(x)).collect()
# calculate execution time
end_time = time.time()
output["customized"]["exe_time"] = end_time - start_time


# output the results
with open(output_path, "w") as output_file:
    json.dump(output, output_file, indent=4)

