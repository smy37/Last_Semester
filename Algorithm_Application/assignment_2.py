import numpy as np
import time
import random
import sys
import matplotlib.pyplot as plt
test_num = 1000
random.seed(0)

test = []
for i in range(test_num):
    temp = []
    temp.append(random.uniform(0,100))
    temp.append(random.uniform(0,100))
    test.append(temp)
# test = sorted(test)

bound = []


start = time.time()
for i in range(len(test)):
    print(i)
    x1, y1 = test[i][0], test[i][1]
    for j in range(i+1, len(test)):
        sensor3 = True
        sensor1, sensor2 = False, False
        x2, y2 = test[j][0], test[j][1]
        a = y1-y2
        b = x2-x1
        c = x1*y2-x2*y1
        for k in range(len(test)):
            if test[k]!=test[i] and test[k]!= test[j]:
                temp = a*test[k][0] + b*test[k][1] + c
                if temp >0:
                    sensor1 = True
                elif temp <0 :
                    sensor2  = True
                else:
                    print(test[i], test[j], test[k])
                    if test[k] < test[i] or test[k] > test[j]:
                        sensor3 = False
                        break
            if sensor1 and sensor2 :
                sensor3 = False
                break
        if sensor3:
            bound.append([test[i], test[j]])



print('소요시간')
print(time.time()-start)

for i in test:
    plt.scatter(i[0], i[1], s= 10, c= 'g')
for i in range(len(bound)):
    plt.plot([bound[i][0][0], bound[i][1][0]], [bound[i][0][1], bound[i][1][1]], c= 'r', linewidth = 3)


plt.show()