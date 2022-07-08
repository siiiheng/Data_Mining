import os
import sys
import random
import time
from blackbox import BlackBox

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

# set random seed
random.seed(553)

# use blackbox
bx = BlackBox()
stream = bx.ask(input_file, stream_size)
stream_user = stream

# count first 100 users
n = len(stream_user)
# create output dict
output = dict()

# create the group for the first 100 users
output[100] = (stream_user[0], stream_user[20], stream_user[40], stream_user[60], stream_user[80])

for i in range(num_ask-1):
    stream = bx.ask(input_file, stream_size)

    for j in stream:
        # for new coming user
        n += 1
        # get a random number
        ran = random.random()
        if ran < 100/n:
            # create a random index
            ind = random.randint(0, 99)
            # replace the previous user with the new one
            stream_user[ind] = j

    # i starts with 0, the next key should be from 200
    output[(i+2)*100] = (stream_user[0], stream_user[20], stream_user[40], stream_user[60], stream_user[80])

# write output to file
with open(output_file, "w+") as file:
    file.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
    for i, j in output.items():
        file.write(str(i) + ",")
        file.write(",".join(j))
        file.write('\n')
file.close()

# end
end = time.time()
print("Execution time:", end-start)


