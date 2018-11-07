# https://blog.csdn.net/quinn1994/article/details/80324308
# 蚁群算法求解TSP
import os

os.getcwd()
# 返回当前工作目录,图片保存在当前目录下
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def getdistmat(coordinates):
    """
    计算两两城市间的欧式距离
    :param coordinates:
    :return:
    """
    num = coordinates.shape[0]
    distmat = np.zeros((num, num))
    for i in range(num):
        for j in range(i, num):  # 由于对称性，只需计算一半数据
            distmat[i][j] = distmat[j][i] = np.linalg.norm(coordinates[i] - coordinates[j])  # 2范数
    return distmat


def cal_transfer_prob(unvisited_list, pheromonetable, current_city, etatable, alpha=1, beta=5):
    """
    计算从当前城市节点到所有未访问城市节点的转移概率
    :param unvisited_list: 未访问城市列表
    :param pheromonetable: 信息素矩阵
    :param current_city: 当前访问的城市
    :param etatable: 启发函数矩阵
    :param alpha: 信息素重要层度因子
    :param beta: 启发函数重要程度因子
    :return:
    """
    probtrans = np.zeros(len(unvisited_list))
    for i, unvisit_city in enumerate(unvisited_list):
        probtrans[i] = (pheromonetable[current_city][unvisit_city] ** alpha) \
                       * (etatable[current_city][unvisit_city] ** beta)

    # eta-从城市i到城市j的启发因子 ,其中pheromonetable[visiting][unvisit]是从本城市到unvisit城市的信息素
    return probtrans / sum(probtrans)


def get_city_by_roulette_selection(probtrans, unvisited_list):
    """
    利用轮赌法得到下一个访问的城市索引
    :param probtrans:
    :param unvisited_list:
    :return:
    """
    # 计算累计概率，斐波那契数列
    cumsum_probtrans = np.cumsum(probtrans)

    # 轮盘赌法选择下一个访问的城市
    k = unvisited_list[list(cumsum_probtrans > np.random.rand()).index(True)]
    return k


def update_phermonetable(pheromonetable, numant, numcity, tabu_list, distmat, rho=0.1, Q=1):
    """

    :param pheromonetable: 信息素矩阵
    :param numant: 蚂蚁的数量
    :param numcity: 城市的数量
    :param tabu_list: 禁忌表
    :param distmat: 距离矩阵
    :param rho: 信息素的挥发率
    :param Q: 完全度，一般为1
    :return:
    """
    change_pheromonetable = np.zeros((numcity, numcity))
    for i in range(numant):  # 更新所有的蚂蚁
        for j in range(numcity - 1):
            start, end = tabu_list[i, j], tabu_list[i, j + 1]  # 蚂蚁从 start 到 end 上释放的信息素
            # 根据公式Q/d更新本只蚂蚁改变的城市间的信息素,其中d是从第j个城市到第j+1个城市的距离
            change_pheromonetable[start][end] += Q / distmat[start][end]
        start, end = tabu_list[i, numcity - 1], tabu_list[i, 0]
        # 最后加入最后一个城市到首城市产生的信息素
        change_pheromonetable[start][end] += Q / distmat[start][end]

    # 信息素更新公式p=(1-挥发速率)*现有信息素+改变的信息素
    pheromonetable = (1 - rho) * pheromonetable + change_pheromonetable
    return pheromonetable


def plot(lengthaver, lengthbest, pathbest, coordinates, numcity):
    """
    绘制算法结果
    :param lengthaver: 每轮平均路径长度
    :param lengthbest: 每轮最优路径长度
    :param pathbest: 每轮最好路径
    :param coordinates: 城市坐标矩阵
    :param numcity: 城市数量
    :return:
    """
    # 做出平均路径长度和最优路径长度
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))  # 绘制带两个子图的图表
    axes[0].plot(lengthaver, 'k', marker='*')
    axes[0].set_title('Average Length')
    axes[0].set_xlabel(u'iteration')

    # 线条颜色black https://blog.csdn.net/ywjun0919/article/details/8692018
    axes[1].plot(lengthbest, 'k', marker='<')
    axes[1].set_title('Best Length')
    axes[1].set_xlabel(u'iteration')
    fig.savefig('Average_Best.png', dpi=500, bbox_inches='tight')
    plt.close()
    fig.show()

    # 作出找到的最优路径图
    bestpath = pathbest[-1]

    plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker='>')  # plot每个城市的坐标点
    plt.xlim([-100, 2000])
    # x范围
    plt.ylim([-100, 1500])
    # y范围
    for i in range(numcity - 1):
        # 按坐标绘出最佳两两城市间路径
        m, n = bestpath[i], bestpath[i + 1]
        print("best_path:", m, n)
        plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')
    print('best path:', bestpath[-1], bestpath[0])

    ## 绘制最后一个城市到第一个城市的路径
    plt.plot([coordinates[int(bestpath[0])][0], coordinates[int(bestpath[51])][0]],
             [coordinates[int(bestpath[0])][1], coordinates[int(bestpath[51])][1]], 'b')

    ax = plt.gca()
    ax.set_title("Best Path")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.savefig('Best Path.png', dpi=500, bbox_inches='tight')
    plt.close()


def main():
    # 初始化城市坐标，总共52个城市
    coordinates = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
                            [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
                            [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
                            [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
                            [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
                            [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
                            [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
                            [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
                            [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
                            [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0],
                            [1340.0, 725.0], [1740.0, 245.0]])

    numant = 40  # 蚂蚁个数
    numcity = coordinates.shape[0]
    # 返回城市距离矩阵
    distmat = getdistmat(coordinates)
    alpha = 1  # 信息素重要程度因子
    beta = 5  # 启发函数重要程度因子
    rho = 0.1  # 信息素的挥发速度
    Q = 1  # 完成率

    max_iter = 150  # 迭代总数

    # diag(),将一维数组转化为方阵 启发函数矩阵，表示蚂蚁从城市i转移到城市j的期望程度
    etatable = 1.0 / (distmat + np.diag([1e10] * numcity))
    # 信息素矩阵
    pheromone_mat = np.ones((numcity, numcity))

    # 禁忌表，可称为路径记录表,记录蚂蚁走过的路径
    tabu_list = np.zeros((numant, numcity), dtype=np.int)

    lengthaver = np.zeros(max_iter)  # 存放每轮迭代后，路径的平均长度
    lengthbest = np.zeros(max_iter)  # 存放每轮迭代后，最优路径长度
    pathbest = np.zeros((max_iter, numcity), dtype=np.int)  # 存放每轮迭代后，最佳路径城市的坐标

    for iter in tqdm(list(range(max_iter)), total=max_iter):
        # 初始化蚁群的第一个城市
        if numant <= numcity:  # 城市数比蚂蚁数多，蚁群初始随机放置于52个城市中
            tabu_list[:, 0] = np.random.permutation(range(numcity))[:numant]
        else:  # 蚂蚁数比城市数多，需要有城市放多个蚂蚁，故每个城市先随机分配一只蚂蚁，再分配剩余的蚂蚁
            tabu_list[:numcity, 0] = np.random.permutation(range(numcity))[:]
            tabu_list[numcity:, 0] = np.random.permutation(range(numcity))[:numant - numcity]

        length = np.zeros(numant)  # 记录每只蚂蚁走过的长度

        # 本段程序算出每只/第i只蚂蚁转移到下一个城市的概率，并访问
        for i in range(numant):

            visiting = tabu_list[i, 0]  # 蚂蚁当前所在的城市(index=0)
            # set()创建一个无序不重复元素集合
            unvisited = set(range(numcity))
            # 未访问的城市集合
            unvisited.remove(visiting)  # 删除已经访问过的城市元素

            for j in range(1, numcity):  # 随机访问完剩余的所有numcity-1个城市

                unvisited_list = list(unvisited)
                # 计算转移概率
                probtrans = cal_transfer_prob(unvisited_list, pheromone_mat, visiting, etatable)

                # 轮盘赌法得到下一个访问的城市坐标
                k = get_city_by_roulette_selection(probtrans, unvisited_list)

                # 访问 k 所在的城市,总长度加入visiting到第k城市的距离
                length[i] += distmat[visiting][k]

                # 更新当前访问的城市
                visiting = k

                # 采用禁忌表来记录蚂蚁i当前走过的第j城市的坐标，这里走了第j个城市.k是下一个访问城市的索引
                tabu_list[i, j] = k

                # 移除已访问的 k
                unvisited.remove(k)

            # 计算本只蚂蚁的总的路径距离，加入最后一个城市到第一个城市的距离
            length[i] += distmat[visiting][tabu_list[i, 0]]

        # 更新信息素
        pheromone_mat = update_phermonetable(pheromone_mat, numant, numcity, tabu_list, distmat)

        # 记录本轮的平均路径长度
        lengthaver[iter] = length.mean()

        # 本部分是为了求出最佳路径

        if iter == 0:
            lengthbest[0] = length.min()
            pathbest[0] = tabu_list[length.argmin()].copy()  # 复制那个走最短路径的蚂蚁对应的禁忌表项
        # 如果是第一轮路径，则选择本轮最短的路径,并返回索值下标，并将其记引录
        else:
            # 后面几轮的情况，更新最佳路径
            if length.min() > lengthbest[iter - 1]:
                lengthbest[iter] = lengthbest[iter - 1]
                pathbest[iter] = pathbest[iter - 1].copy()
            # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
            else:
                lengthbest[iter] = length.min()
                pathbest[iter] = tabu_list[length.argmin()].copy()

    # 迭代完成
    print("ants best length:", lengthbest.min())  # 打印最优路径长度
    # 绘制结果
    plot(lengthaver, lengthbest, pathbest, coordinates, numcity)


if __name__ == '__main__':
    main()
