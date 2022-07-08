from pyspark import SparkContext
import os
import json
import sys

# set local python environment
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# set Vocareum python environment
os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

# set input and output path
input_path = sys.argv[1]
output_path = sys.argv[2]

# read file and create rdd file
sc = SparkContext("local[*]", "task1")
sc.setLogLevel("WARN")
review_rdd = sc.textFile(input_path).map(lambda x: json.loads(x)).cache()

# create dict for final output
output = dict()

# A. The total number of reviews
n_reviews = review_rdd.map(lambda x: (x, 1)).count()

# B. The number of reviews in 2018
n_reviews_2018 = review_rdd.filter(lambda x: x["date"][0:4] == "2018").count()

# C. The number of distinct users who wrote reviews
n_users = review_rdd.map(lambda x: x["user_id"]).distinct().count()

# D. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
top10_users = review_rdd.map(lambda x: (x["user_id"], 1)).reduceByKey(lambda x, y: x+y).\
    sortBy(lambda x: (-x[1], x[0])).take(10)

# E. The number of distinct businesses that have been reviewd
n_businesses = review_rdd.map(lambda x: (x["business_id"], 1)).distinct().count()

# F. The top10 businesses that had the largest numbers of reviews and the number of reviews they had
top10_businesses = review_rdd.map(lambda x: (x["business_id"], 1)).reduceByKey(lambda x, y: x+y).\
    sortBy(lambda x: (-x[1], x[0])).take(10)

# combine all the results
output["n_review"] = n_reviews
output["n_review_2018"] = n_reviews_2018
output["n_user"] = n_users
output["top10_user"] = top10_users
output["n_business"] = n_businesses
output["top10_business"] = top10_businesses

# output the results
with open(output_path, "w") as output_file:
    json.dump(output, output_file, indent=4)
