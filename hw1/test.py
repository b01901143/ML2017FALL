import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import os
import matplotlib.pyplot as plt

input_file = sys.argv[1]
output_file = sys.argv[2]

# print(input_file)
# print(output_file)
# exit()

hour = 9
lambda_w = 1e-5

w = np.load("model_quadratic_all_"+str(hour)+"_regularized_"+str(lambda_w)+".npy")

test_x = []
n_row = 0
# text = open('data/test.csv' ,"r")
text = open(input_file ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])

    for i in range(11-hour,11):
        if r[i] !="NR":
            test_x[n_row//18].append(float(r[i]))
            test_x[n_row//18].append(float(r[i])**2 )
        else:
            test_x[n_row//18].append(0)
            test_x[n_row//18].append(0)
    n_row = n_row+1


# for r in row:
#     if n_row %18 == 9:
#         test_x.ansppend([])
#         for i in range(11-hour,11):
#             if r[i] !="NR":
#                 test_x[n_row//18].append(float(r[i]))
#                 # test_x[n_row//18].append(float(r[i])**2 )
#             else:
#                 test_x[n_row//18].append(0)
#                 # test_x[n_row//18].append(0)
#     n_row = n_row+1
text.close()
test_x = np.array(test_x)

# test_x = (test_x-x_mean[:test_x.shape[0],:])/x_std[:test_x.shape[0],:]

# add square term
# test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = "result/predict_quadratic_all_"+str(hour)+"_regularized_"+str(lambda_w)+".csv"
filename = output_file

dirname = os.path.dirname(filename)
if not os.path.exists(dirname):
    os.makedirs(dirname)
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()

# ans = []
# for i in range(len(test_x)):
#     ans.append(["id_"+str(i)])
#     a = np.dot(w_closed,test_x[i])
#     ans[i].append(a)

# filename = "result/predict_closed_2.csv"
# text = open(filename, "w+")
# s = csv.writer(text,delimiter=',',lineterminator='\n')
# s.writerow(["id","value"])
# for i in range(len(ans)):
#     s.writerow(ans[i]) 
# text.close()
