import matplotlib.pyplot as plt
import numpy as np
x = [1000, 3000, 5000, 7000, 9000]
record = [2.793070793, 24.46092796, 65.25846195, 136.7357121, 236.3218031]



plt.scatter(x, record, c = 'b')


final = []
for i in range(4):
    temp = []
    for j in range(4):
        temp.append((1000*(2*i+1))**(3-j))

    final.append(temp)
final = np.array(final)
print(final)
print(record)
record = np.array(record[:-1])
gae = np.linalg.solve(final, record)
print(gae)
x_t = [i for i in range(1000,20000)]
y_t = []
for i in x_t:
    cnt = 3
    temp =0
    for j in gae:
        temp += j*(i**cnt)
        cnt -=1
    y_t.append(temp)

plt.plot(x_t, y_t, c= 'r')
plt.xlabel('N')
plt.ylabel('Time')
plt.title('Asymptotic plot')
plt.show()