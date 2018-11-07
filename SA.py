import random
import math


def annealingoptimize(domain, costf, T=10000.0, cool=0.95, step=1):
    """
    模拟退火算法
    :param domain:变量的定义域数组
    :param costf: 评价函数
    :param T: 温度
    :param cool: 冷却度
    :param step: 步长
    :return:
    """
    # Initialize the solutions randomly
    vec = [float(random.randint(domain[i][0], domain[i][1], ))
           for i in range(len(domain))]

    while T > 0.1:
        # Choose one of the indices
        i = random.randint(0, len(domain) - 1)

        # Choose a direction to change it
        dir = random.randint(-step, step)

        # Create a new list with one of the values changed
        vecb = vec[:]
        vecb[i] += dir
        if vecb[i] < domain[i][0]:
            vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]:
            vecb[i] = domain[i][1]

        # Calculate the current cost and the new cost
        ea = costf(vec)
        eb = costf(vecb)
        dE = eb - ea
        p = pow(math.e, -dE / T)

        # Is it better, or does it make the probability
        # cutoff?
        if (dE < 0 or p > random.random()):
            vec = vecb  # 接受新解

        # Decrease the temperature
        T = T * cool
        # 若T过大，则搜索到全局最优解的可能会较高，但搜索的过程也就较长。若r过小，则搜索的过程会很快，但最终可能会达到一个局部最优值
    return vec
