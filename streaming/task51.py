import os
import sys
import random
import time
import binascii
from blackbox import BlackBox

def hashpara(x, y, num):
    for i in range(num):
        x.append(random.randint(1, 111111))
        y.append(random.randint(0, 100000))
    return x, y

def myhashs(s):
    result = []
    a = []
    b = []
    a_para, b_para = hashpara(a, b, 2)
    for i in range(2):
        id = int(binascii.hexlify(s.encode('utf8')), 16)
        result.append(((a_para[i] * id + b_para[i]) % 233333) % 69997)
    return result

def check(x):
    for i in x:
        if binarray[i] == 0:
            return 1
    return 0

# start
start = time.time()

# set environment
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Users\\Silvia\\PycharmProjects\\pythonProject1\\venv\\Scripts\\python.EXE"

# input arguments
input_file = sys.argv[1]
stream_size = int(sys.argv[2])
num_ask = int(sys.argv[3])
output_file = sys.argv[4]

# use blackbox
bx = BlackBox()
stream = bx.ask(input_file, stream_size)
# create list for hash values
hash_user = []
for i in stream:
    hash_user.append(myhashs(i))
binarray = [0] * 69997

# align the hash value of the initial users to the array
for i in hash_user:
    for j in i:
        binarray[j] = 1

# create a set for existing users
exist = set(stream)
# create dictionary for output
output = dict()

for i in range(num_ask):
    stream = bx.ask(input_file, stream_size)
    hash_user = []

    # all negative users
    gn = 0
    for j in stream:
        hash_user.append((j, myhashs(j)))
        # find number of negative users
        if j not in exist:
            gn += 1

    # find possible false positive users
    fp_user = []
    # number of real false positive users
    fp = 0

    for j in hash_user:
        if check(j[1]) == 0:
            fp_user.append(j[0])
    for k in fp_user:
        if k not in exist:
            fp += 1

    # deal with the denominator 0
    if gn == 0:
        output[i] = 0.0
    # calculate false positive rate
    else:
        output[i] = fp/gn
    # update the binary array
    for j in hash_user:
        for k in j[1]:
            binarray[k] = 1
    # update the existing users
    exist = exist.union(set(stream))


# write to the output file
with open(output_file, "w+") as file:
    file.write("Time,FPR\n")
    for i, j in output.items():
        file.write(str(i) + "," + str(j))
        file.write('\n')
file.close()

# end
end = time.time()
print("Execution time:", end-start)






