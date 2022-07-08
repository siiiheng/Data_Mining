from pyspark import SparkContext
import time
import os
import sys
import itertools as it
from operator import add
import random

# create dicts for tree structure
def settree(node, matrix):
    child = dict()
    parent = dict()
    vertex = matrix[node]
    for i in vertex:
        if i not in parent.keys():
            parent[i] = set()
            parent[i].add(node)
        else:
            parent[i].add(node)
    child[node] = vertex
    return child, parent, vertex

# create dict for existing items and set their value as 1
def bitvalue(users):
    value = dict()
    for i in users:
        if type(i) is tuple and len(i) == 2:
            value[(i[0], i[1])] = 1
            value[(i[1], i[0])] = 1
        else:
            value[i] = 1
    return value

# Girvan-Newman algorithm
def GN(node, users, matrix):
    # set default level and tree dict
    level = 1
    tree = dict()
    # level 0 has the nodes
    tree[0] = node
    # set a path dict to calculate the edge later
    path = dict()
    # every node has the default value of path (1)
    path[node] = 1

    # get default child dict and parent dict
    child, parent, vertex = settree(node, matrix)

    # create a set for vertex that has been used
    past_ver = set()
    # add first used vertex
    past_ver.add(node)

    while vertex != set():
        # combine current vertex with the used one
        past_ver = vertex.union(past_ver)
        # create a set for new vertex
        new_ver = set()
        for i in vertex:
            child[i] = matrix[i].difference(past_ver)
            # child node contribute to parent node
            for j in child[i]:
                if j not in parent.keys():
                    parent[j] = set()
                    parent[j].add(i)
                else:
                    parent[j].add(i)
            # add path value from parent node
            if parent[i]:
                path[i] = sum([path[k] for k in parent[i]])
            else:
                path[i] = 1
            new_ver = new_ver.union(child[i])

        # add vertex to the current level of the tree
        tree[level] = vertex
        # update the vertex
        vertex = new_ver
        # level up
        level += 1

    # create a dict for edges
    edge = dict()
    # from users create a key-1 dict
    bitmap = bitvalue(users)
    # calculate the edge from the top to bottom
    while level != 1:
        for i in tree[level - 1]:
            for j in parent[i]:
                edge[tuple(sorted((i, j)))] = path[j]/path[i]*bitmap[i]
                bitmap[j] += path[j]/path[i]*bitmap[i]

        level -= 1

    return [(i, j) for i, j in edge.items()]

# find communities
def findcommu(node, users, matrix):
    # create a set for used nodes
    past = set()
    if not matrix[node]:
        past.add(node)
    else:
        past.add(node)
        nodes = [node]
        # add rest nodes to past and nodes list
        while nodes:
            node_1 = nodes[0]
            if len(nodes) == 1:
                nodes = []
            else:
                nodes = nodes[1:]
            for i in matrix[node_1]:
                if i not in past:
                    past.add(i)
                    nodes.append(i)

    # create a list for the communities
    commu = []
    # append single communities to the list
    commu.append(past)
    # check the users remaining except the used ones
    remain = users.difference(past)

    while True:
        # from the remaining users choose a random start
        ran_node = random.sample(remain, 1)[0]
        new_past = set()

        # From this start find a community
        if not matrix[ran_node]:
            new_past.add(ran_node)
        else:
            new_past.add(ran_node)
            new_nodes = [ran_node]
            while new_nodes:
                new_node_1 = new_nodes[0]
                if len(new_nodes) == 1:
                    new_nodes = []
                else:
                    new_nodes = new_nodes[1:]
                for i in matrix[new_node_1]:
                    if i not in new_past:
                        new_past.add(i)
                        new_nodes.append(i)

        # add the used users to previous set
        past = past.union(new_past)
        # append new communities
        commu.append(new_past)
        # update the remaining users
        remain = remain.difference(new_past)

        # break when there is no remaining users
        if not remain:
            break

    return commu

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
# output_b = 'test1.txt'
# output_c = 'test2.txt'

thre = int(sys.argv[1])
input_file = sys.argv[2]
output_b = sys.argv[3]
output_c = sys.argv[4]

sc = SparkContext("local[*]", "task2")
sc.setLogLevel("WARN")

# read file, create rdd
data = sc.textFile(input_file).map(lambda x: x.split(",")).cache()
# find header
header = data.first()
# remove header line from data and get user-business data dictionary
data_noh = data.filter(lambda x: x != header).map(lambda x: (x[0], x[1])).cache()

# create user-business dict
user_busi = data_noh.groupByKey().collectAsMap()

# find distinct users
user_dis = data_noh.map(lambda x: x[0]).distinct().collect()

# filter candidate pairs no less than the threshold
cand = set()
for i in it.combinations(user_dis, 2):
    if len(set(user_busi[i[0]]).intersection(set(user_busi[i[1]]))) >= thre:
        cand.add((i[0], i[1]))

# create set of all users in candidate pairs
cand_user = set()
for i in cand:
    cand_user.add(i[0])
    cand_user.add(i[1])

# create matrix for candidate users
matrix = dict()
for i in cand:
    if i[0] not in matrix.keys():
        matrix[i[0]] = set()
        matrix[i[0]].add(i[1])
    else:
        matrix[i[0]].add(i[1])
    if i[1] not in matrix.keys():
        matrix[i[1]] = set()
        matrix[i[1]].add(i[0])
    else:
        matrix[i[1]].add(i[0])

# calculate the betweenness
bet = sc.parallelize(cand_user).map(lambda x: GN(x, cand_user, matrix)).flatMap(lambda x: [i for i in x])\
    .reduceByKey(add).map(lambda x: (x[0], x[1]/2)).cache()

# sort the results
bet_out = bet.sortBy(lambda x: (-x[1], x[0])).map(lambda x: (x[0], round(x[1], 5))).collect()

# output for betweenness
file_b = open(output_b, "w")
for i in bet_out:
    file_b.write(str(i[0]) + "," + str(i[1]) + "\n")
file_b.close()


# optimize the modularity
# set candidate pair - 1 dict
bitmap = bitvalue(cand)
# set default value for existing edges
remain_edge = len(cand)
# set the default max value of modularity for later comparison
max_mod = -1
# use sorted betweenness
bet_sort = bet.sortBy(lambda x: (-x[1], x[0])).collect()

# create map for number of edges for nodes
lenmap = dict()
for i, j in matrix.items():
    lenmap[i] = len(j)

# calculate modularity and combine nodes to communities
while remain_edge > 0:
    # find top betweenness
    top = bet_sort[0][1]

    # remove the pair from the matrix
    for i in bet_sort:
        if top == i[1]:
            matrix[i[0][0]].remove(i[0][1])
            matrix[i[0][1]].remove(i[0][0])
            # update the number of edges
            remain_edge -= 1
        else:
            break

    # find a random start
    node = random.sample(cand_user, 1)[0]
    # create community list
    commu = findcommu(node, cand_user, matrix)
    # start calculate modularity
    final_mod = 0
    # calculate modularity for the community
    for i in commu:
        mod = 0
        for j in i:
            for k in i:
                if (j, k) not in bitmap.keys():
                    bitmap[(j, k)] = 0
                    mod += bitmap[(j, k)]-lenmap[j]*lenmap[k]/(len(cand)*2)
                else:
                    mod += bitmap[(j, k)]-lenmap[j]*lenmap[k]/(len(cand)*2)
        final_mod += mod

    # calculate the final modularity
    final_mod = final_mod/(len(cand)*2)

    # compare the current modularity with the previous mas value
    # find the real max value and the final best community
    if final_mod > max_mod:
        max_mod = final_mod
        final_commu = commu

    # update the betweenness list as the matrix has been updated
    bet_sort = sc.parallelize(cand_user).map(lambda x: GN(x, cand_user, matrix)).flatMap(lambda x: [i for i in x])\
        .reduceByKey(add).map(lambda x: (x[0], x[1]/2)).sortBy(lambda x: (-x[1], x[0])).collect()


# output for communities
final_commu = sc.parallelize(final_commu).map(lambda x: sorted(x)).sortBy(lambda x: (len(x), x)).collect()

# write the output of communities to the file
file_c = open(output_c, "w")
for i in final_commu:
    file_c.write(str(i)[1:-1])
    file_c.write("\n")
file_c.close()

# end time
end = time.time()
print("Execution time:", end - start)
