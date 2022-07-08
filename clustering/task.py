import sys
import os
import time
import random
from sklearn.cluster import KMeans
import numpy as np
from math import sqrt
import copy
import itertools as it

def getclusnum(x):
    clus_num = dict()
    for i in x:
        if i not in clus_num.keys():
            clus_num[i] = 1
        else:
            clus_num[i] += 1
    return clus_num

def check(x, y):
    id = x[0]
    if x[0] not in y:
        y[id] = dict()
        y[id]["N"] = [data_di_r[tuple(x[1])]]
        y[id]["SUM"] = x[1]
        y[id]["SUMSQ"] = x[1] ** 2
    else:
        y[id]["N"].append(data_di_r[tuple(x[1])])
        y[id]["SUM"] += x[1]
        y[id]["SUMSQ"] += x[1] ** 2

def addpoint(id, point, y):
    y[id]["N"].append(data_di_r[tuple(point)])
    y[id]["SUM"] += point
    y[id]["SUMSQ"] += point ** 2

def findid(distance):
    # all the distance
    dist_value = list(distance.values())
    # find the minimum distance
    dist_min = min(dist_value)
    for i in distance:
        if distance[i] == dist_min:
            id = i
    return id, dist_min

def pointdis(p, cluster):
    n = len(cluster["N"])
    cen = cluster["SUM"] / n
    s = cluster["SUMSQ"] / n - cen ** 2
    x = (p - cen) / s
    d = sqrt(np.dot(x, x))
    return d

def clusdis(c1, c2):
    n1 = len(c1["N"])
    cen1 = c1["SUM"] / n1
    s1 = c1["SUMSQ"] / n1 - cen1 ** 2

    n2 = len(c2["N"])
    cen2 = c2["SUM"] / n2
    s2 = c2["SUMSQ"] / n2 - cen2 ** 2

    x1 = (cen1 - cen2) / s1
    d1 = sqrt(np.dot(x1, x1))

    x2 = (cen1 - cen2) / s2
    d2 = sqrt(np.dot(x2, x2))

    d_final = min(d1, d2)
    return d_final


# start
start = time.time()

# set environment
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# input and output argument
input_file = sys.argv[1]
num_cluster = int(sys.argv[2])
output_file = sys.argv[3]

# read file and load data
with open(input_file) as file:
    data = [i.strip().split(",") for i in file.readlines()]

# create a list for (id, vector) tuple
data_t = list()
for i in data:
    id = int(i[0])
    vec = list()
    for j in i[2:]:
        vec.append(float(j))
    vec = tuple(vec)
    data_t.append((id, vec))

# change tuple list to dict id: vector
data_di = dict(data_t)
# reserve a reversed dict vector: id
value = list(data_di.values())
key = list(data_di.keys())
data_di_r = dict(zip(value, key))

# create an array for vectors
data_arr = list(map(lambda x: np.array(x), value))

# shuffle the data array randomly and use 20% of the total number as sample size
random.shuffle(data_arr)
size = round(len(data_arr) * 0.2)

# write the first line of the output
file = open(output_file, "w")
file.write("The intermediate results:" + "\n")

# step 1: load 20% of the data randomly
sample = data_arr[0:size]

# step 2: run K-Means with a large K on the data in memory using the Euclidean distance as the similarity measurement
k_means = KMeans(n_clusters=num_cluster*25).fit(sample)

# step 3: in the K-Means result, move all the clusters that contain only one point to RS
# outliers
rs = []
# create a dict for cluster with id: number of points
# cluster id is the label in k-means
labels = k_means.labels_
clus_num = getclusnum(labels)

# find the clusters that contain only one point to RS
index_rs = []
for i in clus_num:
    if clus_num[i] == 1:
        for k, v in enumerate(labels):
            if v == i:
                index_rs += [k]
# add the outliers to the rs list
for i in index_rs:
    rs.append(sample[i])
# remove them from the sample
# sorted index list:
index_rs_r = reversed(sorted(index_rs))
for i in index_rs_r:
    sample.pop(i)

# step 4: run K-Means again to cluster the rest of the data points with K = the number of input clusters.
k_means = KMeans(n_clusters=num_cluster).fit(sample)

# step 5: use the K-Means result from Step 4 to generate the DS clusters

# cluster id is the label in k-means
labels = k_means.labels_
# create tuple of id and vectors
clus_t = tuple(zip(labels, sample))
# create a dict for DS with N, SUM, SUMSQ
ds = dict()
for i in clus_t:
    check(i, ds)

# step 6: run K-Means on the points in the RS with a large K to generate CS (clusters with more than one points) and
# RS (clusters with only one point).
if len(rs) == 1:
    k_means = KMeans(n_clusters=len(rs)).fit(rs)
elif len(rs) > 1:
    k_means = KMeans(n_clusters=len(rs)-1).fit(rs)

# create a dict for cluster with id: number of points
# cluster id is the label in k-means
labels = k_means.labels_
clus_num = getclusnum(labels)

# index_clus_rs: index of cluster that only has one point
index_clus_rs = []
for i in clus_num:
    if clus_num[i] == 1:
        index_clus_rs.append(i)
index_rs = []
if len(index_clus_rs) != 0:
    for i in index_clus_rs:
        label_list = list(labels)
        index_rs.append(label_list.index(i))

# create tuple of id and rs
clus_t = tuple(zip(labels, rs))

# create a dict for CS, including N, SUM, SUMSQ
cs = dict()
for i in clus_t:
    if i[0] not in index_clus_rs:
        check(i, cs)

# remove rs points to cs and update the rs data
rs_updated = []
index_rs_r = reversed(sorted(index_rs))
for i in index_rs_r:
    rs_updated.append(rs[i])

# substitute current RS
rs = copy.deepcopy(rs_updated)

# calculate the number of clusters of each type
n_ds = 0
n_cs_clus = len(cs)
n_cs = 0
n_rs = len(rs)

for i in ds:
    n = len(ds[i]["N"])
    n_ds += n
for i in cs:
    n = len(cs[i]["N"])
    n_cs += n

# write to the output
result = []
result.append("Round 1: " + str(n_ds) + "," + str(n_cs_clus) + "," + str(n_cs) + "," + str(n_rs) + "\n")

# for next rounds
for i in range(0, 4):
    # step 7: load another 20% of data randomly
    if i < 3:
        sample = data_arr[(i+1)*size: (i+2)*size]
    else:
        sample = data_arr[size*4:]

    sample_size = len(sample)

    # step 8:  for the new points, compare them to each of the DS using the Mahalanobis Distance and assign
    # them to the nearest DS clusters if the distance is < 2*sqrt(d)

    # create a set for the points already in ds with index
    index_ds = list()

    for j in range(sample_size):
        distance = dict()
        for k in ds:
            distance[k] = pointdis(sample[j], ds[k])

        # find cluster id and minimun distance
        id, dist_min = findid(distance)

        # check if the distance is < 2*sqrt(d)
        d = sqrt(len(sample[j]))
        if dist_min < 2 * d:
            addpoint(id, sample[j], ds)
            index_ds.append(j)

    # step 9: for the new points that are not assigned to DS clusters, using the Mahalanobis Distance and
    # assign the points to the nearest CS clusters if the distance is < 2*sqrt(d)
    if len(cs) != 0:
        # create a set for the points already in cs with index
        index_cs = list()

        for j in range(sample_size):
            if j not in index_ds:
                distance = dict()
                for k in cs:
                    clus_cs = cs[k]
                    distance[k] = pointdis(sample[j], clus_cs)

                # find cluster id and minimum distance
                id, dist_min = findid(distance)

                # check if the distance is < 2*sqrt(d)
                d = sqrt(len(sample[j]))
                if dist_min < 2 * d:
                    addpoint(id, sample[j], cs)
                    index_cs.append(j)

    # step 10: for the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
        try:
            # combine all the index
            index_tt = set(index_ds).union(set(index_cs))
        except NameError:
            index_tt = set(index_ds)
        for j in range(sample_size):
            if j not in index_tt:
                rs.append(sample[j])

    # step 11: run K-Means on the RS with a large K to generate CS and RS
    if len(rs) == 1:
        k_means = KMeans(n_clusters=len(rs)).fit(rs)
    elif len(rs) > 1:
        k_means = KMeans(n_clusters=len(rs)-1).fit(rs)

    # remove all the duplicated labels
    labels_cs = cs.keys()
    labels_cs_updated = set(labels_cs)
    labels_rs = k_means.labels_
    labels_rs_updated = set(labels_rs)

    # find the common and union set of labels
    uni = labels_cs_updated.union(labels_rs_updated)
    com = labels_cs_updated.intersection(labels_rs_updated)

    new = dict()
    for j in com:
        while True:
            n = random.randint(100, sample_size)
            if n not in uni:
                break
        new[j] = n
        uni.add(n)
        # get the new k-means labels
    n_labels = len(list(labels_rs))
    for j in range(n_labels):
        ind = labels_rs[j]
        if ind in new:
            labels_rs[j] = new[ind]

    # create a dict for cluster with id: number of points
    # cluster id is the label in k-means
    clus_num = getclusnum(labels_rs)

    # index_clus_rs: index of cluster that only has one point
    index_clus_rs = []
    for j in clus_num:
        if clus_num[j] == 1:
            index_clus_rs.append(j)
    index_rs = []
    if len(index_clus_rs) != 0:
        for j in index_clus_rs:
            label_list = list(labels_rs)
            index_rs.append(label_list.index(j))

    # add points from RS to CS
    # create tuple of id and rs
    clus_t = tuple(zip(labels_rs, rs))
    for j in clus_t:
        if j[0] not in index_clus_rs:
            check(j, cs)

    # remove rs points to cs and update the rs data
    rs_updated = []
    index_rs_r = reversed(sorted(index_rs))
    for j in index_rs_r:
        rs_updated.append(rs[j])

    # substitute current RS
    rs = copy.deepcopy(rs_updated)

    # step 12: merge CS clusters that have a Mahalanobis Distance < 2*sqrt(d)
    tag = True
    while True:
        merge = list()
        # create list for cs cluster pairs
        clus_id = list(cs.keys())
        clus_id_nodup = set(cs.keys())
        clus_p = list(it.combinations(clus_id, 2))

        for j in clus_p:
            # calculate the distance between clusters
            c1 = j[0]
            c2 = j[1]
            dist_min = clusdis(cs[c1], cs[c2])
            d = sqrt(len(cs[c1]["SUM"]))

            if dist_min < 2 * d:
                cs[c1]["N"] += cs[c2]["N"]
                cs[c1]["SUM"] += cs[c2]["SUM"]
                cs[c1]["SUMSQ"] += cs[c2]["SUMSQ"]
                cs.pop(c2)
                flag = False
                break
        clus_id_updated = set(cs.keys())
        # no merge
        if clus_id_nodup == clus_id_updated:
            break

    # final round, merge DS and CS
    clus_id = list(cs.keys())
    if i == 3 and len(cs) != 0:
        for j in clus_id:
            distance = dict()
            c1 = cs[j]
            for k in ds:
                c2 = ds[k]
                clus_d = clusdis(c1, c2)
                distance[k] = clus_d
            id, dist_min = findid(distance)

            d = sqrt(len(c1["SUM"]))
            if dist_min < 2*d:
                ds[id]["N"] += c1["N"]
                ds[id]["SUM"] += c1["SUM"]
                ds[id]["SUMSQ"] += c1["SUMSQ"]
                cs.pop(j)

    n_ds = 0
    n_cs_clus = len(cs)
    n_cs = 0
    n_rs = len(rs)

    for j in ds:
        n = len(ds[j]["N"])
        n_ds += n
    for j in cs:
        n = len(cs[j]["N"])
        n_cs += n
    result.append("Round " + str(i+2) + ": " + str(n_ds) + "," + str(n_cs_clus) + "," + str(n_cs) + "," + str(n_rs) + "\n")

# add the title line
result.append("\n" + "The clustering results:" + "\n")

# eliminate duplicate items in ds and cs
for i in ds:
    ds[i]["N"] = set(ds[i]["N"])
if len(cs) != 0:
    for i in cs:
        cs[i]["N"] = set(cs[i]["N"])

# collect all rs points
rs_set = set()
for i in rs:
    rs_set.add(data_di_r[tuple(i)])

# check all points and add the result to output
index_tt = len(data_di)
for i in range(index_tt):
    if i in rs_set:
        result.append(str(i) + ",-1" + "\n")
    else:
        for j in cs:
            if i in cs[j]["N"]:
                result.append(str(i) + ",-1" + "\n")
                break

        for j in ds:
            if i in ds[j]["N"]:
                result.append(str(i) + "," + str(j) + "\n")
                break


# write to output file
for i in result:
    file.write(i)

# end
end = time.time()
print("Execution time: ", end-start)
