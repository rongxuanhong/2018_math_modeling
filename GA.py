import random
import numpy as np
from scipy.optimize import fsolve

population_size = 10  # population size
pc = 0.6  # probability of crossover
pm = 0.01  # probability of mutation
boundary_list = [[-3.0, 12.1], [4.1, 5.8]]


# https://blog.csdn.net/qq_30666517/article/details/78637255
# https://blog.csdn.net/longxinghaofeng/article/details/77504212


## uint8:0-255
def b2d(b, length):
    """
    二进制转十进制
    :param b: 二进制数
    :param length: 二进制编码的长度
    :return:
    """
    decimal = 0
    for j in range(len(b)):
        decimal += b[j] * (2 ** (length - 1 - j))
    return decimal


def getEncodeLength(boundary_list, delta=0.0001):
    """
    得到所有变量的编码的长度集合
    :param boundary_list: 变量的区间列表
    :param delta: 变量的精度要求
    :return:
    """
    # 每个变量的编码长度
    lengths = []
    for boundary in boundary_list:
        lower = boundary[0]
        upper = boundary[1]
        # lamnda 代表匿名函数f(x)=0,50代表搜索的初始解 求解一元方程得解length
        res = fsolve(lambda x: ((upper - lower) * 1 / delta) - 2 ** x - 1, np.array([50]))
        length = int(res[0])
        lengths.append(length)
    return lengths


def d2value(d, boundary, length):
    """
    十进制数转到决策变量所在区间内的数
    :param d:十进制数
    :param boundary: 决策变量的上下界
    :param length: 决策变量编码后的基因长度
    :return:
    """
    lower = boundary[0]
    upper = boundary[1]
    interval = upper - lower  ##  当前变量的区间
    value = lower + (d * interval) / (2 ** length - 1)
    return value


def geneEncoding(pop_size, encodeLengths):
    """
    初始化种群，采用随机编码 0/1
    :param pop_size:
    :param encodeLengths:
    :return:
    """
    pop = np.zeros((pop_size, sum(encodeLengths)), dtype=np.uint8)
    for i in range(pop_size):
        pop[i, :] = np.random.randint(0, 2, sum(encodeLengths))
    return pop


def decode_chrom(pop, encodeLengths, boundary_list):
    """
    对编码的种群进行解码
    :param pop:
    :param encodeLengths:
    :param boundary_list:
    :return:
    """
    pop_decoded = np.zeros((pop.shape[0], len(encodeLengths)))  # population decoded
    for k, individual in enumerate(pop):
        start = 0
        for i, length in enumerate(encodeLengths):  ## 对每条染色体上的各个决策变量组成的基因进行解码
            boundary = (boundary_list[i][0], boundary_list[i][1])
            pop_decoded[k, i] = d2value(b2d(individual[start:start + length], length), boundary, length)
            start = length
    return pop_decoded


def getFitnessValue(func, pop_decoded):
    """
    得到种群的每个个体的适应度值及每个个体被选择的累积概率
    :param func:
    :param pop_decoded:
    :return:
    """
    # 初始化种群的适应度值为0
    fitness_values = np.zeros((pop_decoded.shape[0], 1))
    # 计算适应度值
    for i, individual in enumerate(pop_decoded):
        fitness_values[i, 0] = func(individual)  # 计算每个个体的适应度值
    # 计算每个染色体被选择的概率
    probability = fitness_values / np.sum(fitness_values)
    # 得到每个染色体被选中的累积概率
    cum_probability = np.cumsum(probability)
    return fitness_values, cum_probability


# 新种群选择
def selectNewPopulation(chromosomes, cum_probability):
    """
    轮盘赌法选择留下的个体构成新种群
    :param chromosomes: 种群:染色体集合
    :param cum_probability:
    :return:
    """
    pop_size, n_gene = chromosomes.shape  # 种群大小，基因个数
    new_population = np.zeros((pop_size, n_gene), dtype=np.uint8)
    # 随机产生M个概率值:相当于转动轮盘 M 次的结果
    randoms = np.random.rand(pop_size)
    index = 0
    for i, random in enumerate(randoms):
        # logical = cum_probability >= random
        # index = np.where(logical == 1)
        # # index是tuple,tuple中元素是ndarray
        # newpopulation[i, :] = chromosomes[index[0][0], :]
        if random < cum_probability[i]:
            new_population[index, :] = chromosomes[i, :]
            index += 1

    return new_population


def crossover(population, Pc=0.8):
    """
    :param population: 新种群
    :param Pc: 交叉概率默认是0.8
    :return: 交叉后得到的新种群
    """
    # 根据交叉概率计算需要进行交叉的个体个数
    pop_size, n_gene = population.shape
    numbers = np.uint8(pop_size * Pc)
    # 确保进行交叉的染色体个数是偶数个
    if int(numbers) % 2 != 0:
        numbers += 1
    # 复制原种群
    update_population = np.copy(population)
    # 抽取出number个染色体索引
    indexs = random.sample(range(pop_size), numbers)

    # 对复制后的种群进行交叉分配
    while len(indexs) > 0:
        a = indexs.pop()
        b = indexs.pop()
        # 随机产生一个交叉点
        crossoverPoint = random.sample(range(1, n_gene), 1)[0]
        # one-single-point crossover 两条染色体的后段交换
        update_population[a, crossoverPoint:] = population[b, crossoverPoint:]
        update_population[b, crossoverPoint:] = population[a, crossoverPoint:]

    return update_population


# 染色体变异
def mutation(population, Pm=0.01):
    """
    对种群中某些基因进行变异
    :param population: 经交叉后得到的种群
    :param Pm: 变异概率默认是0.01
    :return: 经变异操作后的新种群
    """
    update_population = np.copy(population)
    pop_size, n_gene = population.shape
    # 计算需要变异的基因个数
    total_genes = pop_size * n_gene
    genes = np.uint8(total_genes * Pm)
    # 将所有的基因按照序号进行10进制编码，则共有total_genes个基因
    # 随机抽取(采样)genes个基因进行基本位变异
    mutationGeneIndexs = random.sample(range(0, total_genes), genes)
    # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)
    for gene in mutationGeneIndexs:
        # 确定变异基因位于第几个染色体
        chromosomeIndex = gene // n_gene
        # 确定变异基因位于当前染色体的第几个基因位
        geneIndex = gene % n_gene
        # mutation
        if update_population[chromosomeIndex, geneIndex] == 0:
            update_population[chromosomeIndex, geneIndex] = 1
        else:
            update_population[chromosomeIndex, geneIndex] = 0
    return update_population


# 定义适应度函数
def fitnessFunction():
    return lambda x: 21.5 + x[0] * np.sin(4 * np.pi * x[0]) + x[1] * np.sin(20 * np.pi * x[1])


def main(max_iter=500):
    """
    500个种均只群繁衍1次,种群多样性提高
    :param max_iter:
    :return:
    """
    # 每次迭代得到的最优解
    optimalSolutions = []
    optimalValues = []
    # 决策变量的取值范围
    decisionVariables = [[-3.0, 12.1], [4.1, 5.8]]
    population_size = 10
    # 得到染色体编码长度

    lengthsEncode = getEncodeLength(decisionVariables)
    # 得到初始种群编码
    chromosomesEncoded = geneEncoding(population_size, lengthsEncode)
    # 种群解码
    decoded = decode_chrom(chromosomesEncoded, lengthsEncode, decisionVariables)
    for iteration in range(max_iter):
        # 得到个体适应度值和个体的累积概率
        evalvalues, cum_proba = getFitnessValue(fitnessFunction(), decoded)
        # 选择新的种群
        newpopulations = selectNewPopulation(chromosomesEncoded, cum_proba)
        # 进行交叉操作
        crossoverpopulation = crossover(newpopulations)
        # mutation
        mutationpopulation = mutation(crossoverpopulation)
        # 将变异后的种群解码，得到每轮迭代最终的种群
        final_decoded = decode_chrom(mutationpopulation, lengthsEncode, decisionVariables)
        # 适应度评价
        fitnessvalues, cum_individual_proba = getFitnessValue(fitnessFunction(), final_decoded)
        # 搜索每次迭代的最优解，以及最优解对应的目标函数的取值
        optimalValues.append(np.max(list(fitnessvalues)))  # 取适应度最大的值
        index = np.where(fitnessvalues == max(list(fitnessvalues)))
        optimalSolutions.append(final_decoded[index[0][0], :])  # 取适应度最大值对应的个体
        decoded = final_decoded
    # 搜索最优解
    optimalValue = np.max(optimalValues)
    optimalIndex = np.where(optimalValues == optimalValue)
    optimalSolution = optimalSolutions[optimalIndex[0][0]]
    return optimalSolution, optimalValue


def main2(max_iter=500):
    """
    10个种群繁衍500次，新种群为上次迭代选择交叉变异后的种群
    :param max_iter:
    :return:
    """
    # 每次迭代得到的最优解
    optimalSolutions = []
    optimalValues = []
    # 决策变量的取值范围
    decisionVariables = [[-3.0, 12.1], [4.1, 5.8]]
    population_size = 10
    # 得到染色体编码长度

    lengthsEncode = getEncodeLength(decisionVariables)
    # 得到初始种群编码
    chromosomesEncoded = geneEncoding(population_size, lengthsEncode)
    # 种群解码
    decoded = decode_chrom(chromosomesEncoded, lengthsEncode, decisionVariables)
    for iteration in range(max_iter):
        # 得到个体适应度值和个体的累积概率
        evalvalues, cum_proba = getFitnessValue(fitnessFunction(), decoded)
        # 选择新的种群
        newpopulations = selectNewPopulation(chromosomesEncoded, cum_proba)
        # 进行交叉操作
        crossoverpopulation = crossover(newpopulations)
        # mutation
        mutationpopulation = mutation(crossoverpopulation)
        # 将变异后的种群解码，得到每轮迭代最终的种群
        final_decoded = decode_chrom(mutationpopulation, lengthsEncode, decisionVariables)
        # 适应度评价
        fitnessvalues, cum_individual_proba = getFitnessValue(fitnessFunction(), final_decoded)
        # 搜索每次迭代的最优解，以及最优解对应的目标函数的取值
        optimalValues.append(np.max(list(fitnessvalues)))  # 取适应度最大的值
        index = np.where(fitnessvalues == max(list(fitnessvalues)))
        optimalSolutions.append(final_decoded[index[0][0], :])  # 取适应度最大值对应的个体
        decoded = final_decoded
    # 搜索最优解
    optimalValue = np.max(optimalValues)
    optimalIndex = np.where(optimalValues == optimalValue)
    optimalSolution = optimalSolutions[optimalIndex[0][0]]
    return optimalSolution, optimalValue


if __name__ == '__main__':
    import timeit

    solution, value = main()
    print('最优解: x1, x2')
    print(solution[0], solution[1])
    print('最优目标函数值:', value)
    # 测量运行时间
    elapsedtime = timeit.timeit(stmt=main, number=1)
    print('Searching Time Elapsed:(S)', elapsedtime)
