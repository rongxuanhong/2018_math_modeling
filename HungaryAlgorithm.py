import timeit
from collections import deque

# ==============================================================================
# 匈牙利算法 https://luzhijun.github.io/2016/10/10/%E5%8C%88%E7%89%99%E5%88%A9%E7%AE%97%E6%B3%95%E8%AF%A6%E8%A7%A3/
# https://blog.csdn.net/dark_scope/article/details/8880547
# 只能求出某个最大匹配 但不能求出最小代价的最大匹配
# ==============================================================================
class HungarianAlgorithm(object):
    def __init__(self, graph):
        """
        @graph:图的矩阵表示
        """
        self.graph = graph
        self.n = len(graph)

    def find(self, x):
        for i in range(self.n):
            if self.graph[x][i] != 0 and not self.used[i]:  # x到i有连接，
                self.used[i] = 1  # 放入交替路
                if self.match[i] == -1 or self.find(self.match[i]) == 1:  # 如果i还加入匹配或者i的原匹配点可以继续通过增加增广路径找到新的匹配
                    # i，x加入匹配 匹配边为 x->i
                    self.match[i] = x
                    self.match[x] = i
                    print(x + 1, '->', i + 1)
                    return 1  # 找到一条匹配
        return 0  # 未找到一条匹配

    def hungarian1(self):
        """递归形式
        """
        self.match = [-1] * self.n  # 记录匹配情况 -1表示未匹配
        m = 0
        cost = 0
        for i in range(self.n):
            if self.match[i] == -1:
                self.used = [False] * self.n  # 记录是否访问过 是否在当前的增广路径中
                print('开始匹配:', i + 1)
                m += self.find(i)
        return m

    def hungarian2(self):
        """循环形式
        """
        match = [-1] * self.n  # 记录匹配情况
        used = [-1] * self.n  # 记录是否访问过
        Q = deque()  # 设置队列
        ans = 0
        prev = [0] * self.n  # 代表上一节点
        for i in range(self.n):
            if match[i] == -1:
                Q.clear()
                Q.append(i)
                prev[i] = -1  # 设i为出发点
                flag = False  # 未找到增广路
                while len(Q) > 0 and not flag:
                    u = Q.popleft()
                    for j in range(self.n):
                        if not flag and self.graph[u][j] == 1 and used[j] != i:
                            used[j] = i
                            if match[j] != -1:
                                Q.append(match[j])
                                prev[match[j]] = u  # 记录点的顺序
                            else:
                                flag = True
                                d = u
                                e = j
                                while (d != -1):  # 将原匹配的边去掉加入原来不在匹配中的边
                                    t = match[d]
                                    match[d] = e
                                    match[e] = d
                                    d = prev[d]
                                    e = t
                                print('mathch:', match)
                                print('prev:', prev)
                                print('deque', Q)
                if match[i] != -1:  # 新增匹配边
                    ans += 1
        return ans


def do1():
    graph = [(0, 0, 0, 0, 1, 0, 1, 0),
             (0, 0, 0, 0, 1, 0, 0, 0),
             (0, 0, 0, 0, 1, 1, 0, 0),
             (0, 0, 0, 0, 0, 0, 1, 1),
             (1, 1, 1, 0, 0, 0, 0, 0),
             (0, 0, 1, 0, 0, 0, 0, 0),
             (1, 0, 0, 1, 0, 0, 0, 0),
             (0, 0, 0, 1, 0, 0, 0, 0)]
    h = HungarianAlgorithm(graph)
    print(h.hungarian1())


def do2():
    graph = [(0, 0, 0, 0, 1, 0, 1, 0),
             (0, 0, 0, 0, 1, 0, 0, 0),
             (0, 0, 0, 0, 1, 1, 0, 0),
             (0, 0, 0, 0, 0, 0, 1, 1),
             (1, 1, 1, 0, 0, 0, 0, 0),
             (0, 0, 1, 0, 0, 0, 0, 0),
             (1, 0, 0, 1, 0, 0, 0, 0),
             (0, 0, 0, 1, 0, 0, 0, 0)]
    h = HungarianAlgorithm(graph)
    print(h.hungarian2())


if __name__ == '__main__':
    do1()
