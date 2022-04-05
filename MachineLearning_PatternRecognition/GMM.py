import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from scipy.stats import norm, multivariate_normal


problem = 'p2'
iris = load_iris()
data_path = r'C:\Users\seong\Desktop\2022_1학기\기계학습및패턴인식\hw1_data_win'
t_d = open(data_path+f'/{problem}_train_input_win.txt').readlines()
t_l = open(data_path+f'/{problem}_train_target_win.txt').readlines()
a_d = open(data_path+f'/{problem}_test_input_win.txt').readlines()
a_l = open(data_path+f'/{problem}_test_target_win.txt').readlines()


##### 2-1. Train Data를 전처리!!
train_data = []
for i in t_d:
    temp = []
    tt = i.strip().replace('\t', ' ').split(' ')
    for j in tt:
        if j  != '':
            temp.append(j)
    train_data.append(list(map(float,temp)))
train_data = np.array(train_data)


##### 2-2. Train Label을 전처리!!
train_label = []
for i in t_l: ###### 여기를 고치면 들어가는 데이터가 바뀜
    temp = []
    tt = i.strip().replace('\t', ' ').split(' ')
    for j in tt:
        if j != '':
            temp.append(j)
    train_label.append(list(map(int, temp)))
train_label = np.array(train_label)

##### 2-3. Test Data를 전처리!!
test_data = []
for i in a_d:
    temp = []
    tt = i.strip().replace('\t', ' ').split(' ')
    for j in tt:
        if j  != '':
            temp.append(j)
    test_data.append(list(map(float,temp)))
test_data = np.array(test_data)

##### 2-4. Test Label을 전처리!!
test_label = []
for i in a_l: ###### 여기를 고치면 들어가는 데이터가 바뀜
    temp = []
    tt = i.strip().replace('\t', ' ').split(' ')
    for j in tt:
        if j != '':
            temp.append(j)
    test_label.append(list(map(int, temp)))
test_label = np.array(test_label)



##### 3. 산점도 표시.
plot_object = train_data
plot_target = train_label
c_1 = []
c_2 = []
for i in range(len(plot_object)):
    if plot_target[i] == 0:
        c_1.append(plot_object[i])

    elif plot_target[i] == 1:
        c_2.append(plot_object[i])
#
# plt.scatter([i[0] for i in c_1], [i[1] for i in c_1], c= 'b', label = 'Class1')
# plt.scatter([i[0] for i in c_2], [i[1] for i in c_2], c= 'r', label = 'Class2')
#
# plt.legend()
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Training Data Scatter Plot')
# plt.savefig('./result/data_test.png')
# plt.show()

def get_score(prediction, answer):
    result = 0
    for i in range(len(answer)):
        if prediction[i] == answer[i][0]:
            result +=1
    return result, (result/len(answer))*100

def plot_contours(data1, data2, means, covs, title, name):
    """visualize the gaussian components over the data"""
    plt.figure()
    print([i[0] for i in data1])
    plt.scatter([i[0] for i in data1], [i[1] for i in data1], c= 'b', label = 'Class1')
    plt.scatter([i[0] for i in data2], [i[1] for i in data2], c= 'r', label = 'Class2')

    delta = 0.0055555556
    k = means.shape[0]
    x = np.arange(-1.0, 1.0, delta)
    y = np.arange(-1.0, 1.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid)
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(name + ' Graph')
    plt.tight_layout()
    plt.savefig(f'./result2/2_{name}.png')
    plt.cla()
    plt.clf()
    plt.close()


cov_type = ['spherical', 'diag', 'full']
component_numbers= [i for i in range(1,10)]
tol_list = [1e-3, 1e-4, 1e-5]

#
f = open('./result2/result.csv','w', newline='')
wr = csv.writer(f)
wr.writerow(['Status', 'Cov Matrix', 'Threshold', 'Component Number', 'Correct Num', 'Accuracy', 'Converge Step'])

for ct in cov_type:
    for thr in tol_list:
        for cn in component_numbers:
            gmm = GaussianMixture(n_components=cn, covariance_type=ct, max_iter = 1000, tol = thr,  random_state = 4).fit(train_data)
            gmm_cluster_labels_t = gmm.predict(train_data)
            gmm_cluster_labels_v = gmm.predict(test_data)

            name = ct+'_'+str(thr)+'_'+str(cn)
            plot_contours(c_1, c_2, gmm.means_, gmm.covariances_, 'Initial clusters', name)
            t_c, t_a = get_score(gmm_cluster_labels_t, train_label)
            v_c, v_a = get_score(gmm_cluster_labels_v, test_label)
            print(f'{ct} / {thr} / {cn}개, Training Result: 맞춘개수 = {t_c}, 정확도 = {t_a}, 수렴 지점 = {gmm.n_iter_}, 수렴 여부 = {gmm.converged_}')
            print(f'{ct} / {thr} / {cn}개, Test Result: 맞춘개수 = {v_c}, 정확도 = {v_a}')
            wr.writerow(['train',ct, thr, cn, t_c, t_a, gmm.n_iter_])
            wr.writerow(['test',ct, thr, cn, v_c, v_a, gmm.n_iter_])
            print()