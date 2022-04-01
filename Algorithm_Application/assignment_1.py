from pprint import pprint
import sys


def hanoi_v2(n_list, start, destination, mid):
    if len(n_list) == 1 and destination != 'B' and start != 'B':
        return [(n_list[0], start, mid), (n_list[0], mid, destination)]
    elif len(n_list) ==1 and destination == 'B':
        return [(n_list[0], start, mid)]
    elif len(n_list) == 1 and start == 'B':
        return [(n_list[0], start, destination)]
    else:
        return hanoi_v2(n_list[:-1], start, destination, mid) + hanoi_v2([n_list[-1]], start, mid, mid)\
               + hanoi_v2(n_list[:-1], destination, start,  mid)+ hanoi_v2([n_list[-1]], mid, destination, mid)\
                + hanoi_v2(n_list[:-1], start, destination, mid)


def combination(n, k, n_list, stack):
    if n==k:
        return [n_list+stack]
    elif k==0:
        return [stack]
    else:
        return combination(n-1, k, n_list[:-1], stack) + combination(n-1, k-1, n_list[:-1], stack+[n_list[-1]])


def newton(num, ans, tol):
    if abs(ans**2-num) <=tol:
        return ans
    else:
        return newton(num, (ans**2+num)/(2*ans),tol)



if __name__ =='__main__':
    tol = 0.000001
    num = int(input())
    start_ans = round(num/2)
    print(newton(num, start_ans, tol))






    # n, k = list(map(int, sys.stdin.readline().strip().split(' ')))
    # n_list = [x+1 for x in range(n)]
    # temp = combination(n, k, n_list, [])
    # print(f'조합의 개수: {len(temp)}')
    # pprint(f'조합 양상: {temp}')



    # n_list = [x+1 for x in range(n)]
    # result = hanoi_v2(n_list, 'A', 'C', 'B')
    # print(f'전체 이동 횟수: {len(result)}')
    # pprint(f'전체 이동 양상: {result}')

