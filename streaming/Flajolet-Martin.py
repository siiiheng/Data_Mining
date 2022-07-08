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

# find zero
def findzero(x):
    len_x = len(bin(x))
    len_x_nozero = len(bin(x).rstrip("0"))
    num_zero = len_x - len_x_nozero
    return  num_zero

def myhashs(s):
    result = []
    a = []
    b = []
    a_para, b_para = hashpara(a, b, n_hash)
    for i in range(n_hash):
        id = int(binascii.hexlify(s.encode('utf8')), 16)
        value = ((a_para[i] * id + b_para[i]) % 2333) % 69997
        result.append(findzero(value))
    return result

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

# set number of hash functions
n_hash = 200

# use blackbox
bx = BlackBox()
stream = bx.ask(input_file, stream_size)
output = dict()

for i in range(num_ask):
    stream = bx.ask(input_file, stream_size)
    # create list for hash values
    hashlist = []
    for j in stream:
        hashlist.append(myhashs(j))

    # count the number of zeros
    num_zero = []
    for j in range(n_hash):
        binlist = []
        for k in hashlist:
            binlist.append(k[j])
        # add the max number of zeros for each hashlist item to the list
        num_zero.append(max(binlist))

    # calculate the window
    num_user = len(set(stream))
    # create window list
    win = []
    # add number of windows of different scales to the window list
    for j in num_zero:
        win.append(2**j)
    # calculate the window number
    num_win = sum(win)/n_hash

    # assign to the output
    output[i] = (num_user, int(num_win))

# write to the output file
with open(output_file, "w+") as file:
    file.write("Time,Ground Truth,Estimation\n")
    for i, j in output.items():
        file.write(str(i) + "," + str(j[0]) + "," + str(j[1]))
        file.write('\n')
file.close()

# end
end = time.time()
print("Execution time:", end-start)


