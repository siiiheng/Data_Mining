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

# set input and output path
review_file = sys.argv[1]
business_file = sys.argv[2]
output_a = sys.argv[3]
output_b = sys.argv[4]

# review_file = 'test_review.json'
# business_file = 'business.json'
# output_a = 'output3a.txt'
# output_b = 'output3b.json'

# read file and create rdd file
sc = SparkContext("local[*]", "task3")
sc.setLogLevel("WARN")
review_rdd = sc.textFile(review_file).map(lambda x: json.loads(x))
business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x))

# A. average star for each city
# map business and city data
city_rdd = business_rdd.map(lambda x: (x["business_id"], x["city"]))
# map business and star data
star_rdd = review_rdd.map(lambda x: (x["business_id"], x["stars"]))


# create a function to change None value of city to ""
def modify_none(x):
    if x[0] is None:
        x[0] = ""
    return x


# join two rdd by business id, modify none values, and calculate average star
join_rdd = city_rdd.leftOuterJoin(star_rdd).\
    map(lambda x: modify_none(x[1])).\
    filter(lambda x: x[1] is not None).\
    groupByKey().map(lambda x: (x[0], sum(x[1]) / len(x[1]))).cache()

# sorting
city_list = join_rdd.sortBy(lambda x: (-x[1], x[0])).collect()

with open(output_a, "w") as output_a:
    output_a.write("city,stars\n")
    output_star = [str(x[0]) + "," + str(x[1]) + "\n" for x in city_list]
    output_a.writelines(output_star)

# B compare Python and Spark
output = dict()

# Python
start_time = time.time()

# repeat 3A
review_rdd = sc.textFile(review_file).map(lambda x: json.loads(x))
business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x))
city_rdd = business_rdd.map(lambda x: (x["business_id"], x["city"]))
star_rdd = review_rdd.map(lambda x: (x["business_id"], x["stars"]))


def modify_none(x):
    if x[0] is None:
        x[0] = ""
    return x


join_rdd = city_rdd.leftOuterJoin(star_rdd).map(lambda x: modify_none(x[1])).\
    filter(lambda x: x[1] is not None).groupByKey().map(lambda x: (x[0], sum(x[1]) / len(x[1]))).cache()

# sorting
sorted_city = sorted(join_rdd.collect(), key=lambda x: (-x[1], x[0]))

if len(sorted_city) >= 10:
    top10_city = sorted_city[:10]
else:
    top10_city = sorted_city

print(top10_city)

end_time = time.time()
output["m1"] = end_time - start_time

# 2. Spark
start_time = time.time()

# repeat 3A
review_rdd = sc.textFile(review_file).map(lambda x: json.loads(x))
business_rdd = sc.textFile(business_file).map(lambda x: json.loads(x))
city_rdd = business_rdd.map(lambda x: (x["business_id"], x["city"]))
star_rdd = review_rdd.map(lambda x: (x["business_id"], x["stars"]))


def modify_none(x):
    if x[0] is None:
        x[0] = ""
    return x


join_rdd = city_rdd.leftOuterJoin(star_rdd).map(lambda x: modify_none(x[1])).\
    filter(lambda x: x[1] is not None).groupByKey().map(lambda x: (x[0], sum(x[1]) / len(x[1]))).cache()

# sorting
top10_city = join_rdd.sortBy(lambda x: (-x[1], x[0])).take(10)

print(top10_city)

end_time = time.time()
output["m2"] = end_time - start_time
output["reason"] = "Python is a little bit faster because: 1. Python collects all data at once and process all, " \
                   "while Spark aggregates data from distributed chunks and does the sorting in parallel. " \
                   "2. The transformation of Spark are lazy and when we do the actions, the previous transformations " \
                   "will be operated. The map tasks and reduce tasks cost time. However, sorting in Python " \
                   "doesn't need to spend extra time on these transformations."

with open(output_b, "w") as output_file:
    json.dump(output, output_file, indent=4)
