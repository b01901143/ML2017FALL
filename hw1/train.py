import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import matplotlib.pyplot as plt

#feature scaling
#adagrade
#N-fold cross validation
#batch

data = []
hour = 9
lambda_w = 0.0001

for i in range(18):
    data.append([])

n_row = 0
text = open('data/train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0)) 
    n_row = n_row+1
text.close()

x = []
y = []

for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(480-hour):
        x.append([])
        # 18種污染物
        for t in range(18):
            # if t!=9:
            #     continue
            # 連續9小時
            for s in range(hour):
                x[(480-hour)*i+j].append(data[t][480*i+j+s] )
                # x[(480-hour)*i+j].append(data[t][480*i+j+s]**2 )
        y.append(data[9][480*i+j+hour])
x = np.array(x)
y = np.array(y)

print(x.shape)
# x_mean = np.reshape(np.repeat(x.mean(axis=0),x.shape[0]),x.shape)    
# x_std = np.reshape(np.repeat(x.std(axis=0),x.shape[0]),x.shape)  
# x = (x-x_mean)/x_std

# add square term
# x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

w = np.zeros(len(x[0]))
l_rate = 10
repeat = int(1e6)

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

costs = []
# pre_cost = 1e6

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = np.sum(loss**2 )/ len(x) + lambda_w*np.sum(w**2) 
    # if cost > pre_cost and i>repeat//100:
    #     break
    # pre_cost = cost
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss) + lambda_w*w*len(x)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
    costs.append(cost_a)

#use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh 
w_closed = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)


# save model
np.save("model_linear_all_"+str(hour)+"_regularized_"+str(lambda_w)+".npy",w)
# read model


plt.xlabel('epoch', fontsize = 18)
plt.ylabel('cost', fontsize = 18)
plt.axis([0, len(costs), 4, 10])

d1p, = plt.plot([i for i in range(len(costs))], costs, linewidth=0.5, color='g', markersize=0.3)
plt.legend([d1p], ["traing cost = "+str(round(costs[-1],6) )])

plt.savefig('result/traing_curve_linear_all_'+str(hour)+"_regularized_"+str(lambda_w)+'.png', format='png', dpi=1000)
# plt.show()