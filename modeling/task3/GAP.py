import random
import numpy as np

####################### 遗传算法 ###########################
def geneEncoding(pop_size, chromlength):
    """
    初始化种群
    :param pop_size:
    :param chromlength:
    :return:
    """
    pop = (np.random.rand(pop_size, chromlength) * 2 - 1) + (np.random.rand(pop_size, chromlength) * 2 - 1) * 1j
    return pop


def getFitnessValue(pop):
    """
    得到种群的每个个体的适应度值及每个个体被选择的累积概率
    :param pop:
    :return:
    """
    # 初始化种群的适应度值为0
    pop_size, len = pop.shape
    fitness_values = np.zeros((pop.shape[0], 1))
    dismat = np.full((len, len), np.inf)
    # 计算适应度值
    for k, individual in enumerate(pop):
        # fitness_values[i, 0] = func(individual)  # 计算每个个体的适应度值
        for i in range(8):
            for j in range(i + 1, 8):
                dismat[i][j] = abs(individual[i] - individual[j])
        min_dis = np.min(dismat)
        if min_dis:
            # pop[k, :] = pop[k, :] / min_dis
            fitness_values[k] = 1 / np.mean(abs((pop[k, :] / min_dis) ** 2))
        else:
            fitness_values[k] = 0

    # 计算每个染色体被选择的概率
    probability = fitness_values / np.sum(fitness_values)
    # 得到每个染色体被选中的累积概率
    cum_probability = np.cumsum(probability)
    return fitness_values, cum_probability


# 新种群选择
def selectNewPopulation(chromosomes, cum_probability, fitness_Value):
    """
    轮盘赌法选择留下的个体构成新种群
    :param chromosomes: 种群:染色体集合
    :param cum_probability:
    :return:
    """
    pop_size, n_gene = chromosomes.shape  # 种群大小，基因个数
    # new_population = np.zeros((pop_size, n_gene), dtype=np.uint8)
    # 随机产生M个概率值:相当于转动轮盘 M 次的结果
    randoms = np.random.rand(pop_size)
    index = 0
    data = list()
    fitnessValue = list()
    for i, random in enumerate(randoms):

        if random < cum_probability[i]:
            # new_population[index, :] = chromosomes[i, :]
            index += 1
            data.append(chromosomes[i, :])
            fitnessValue.append(fitness_Value[i])
    new_population, fitnessValue = np.stack(data), np.stack(fitnessValue)
    return fitnessValue, new_population,


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
def mutation(population, chromlength, Pm=0.01):
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
    # # 将所有的基因按照序号进行10进制编码，则共有total_genes个基因
    # # 随机抽取(采样)genes个基因进行基本位变异
    mutationGeneIndexs = random.sample(range(0, total_genes), genes)
    # # 确定每个将要变异的基因在整个染色体中的基因座(即基因的具体位置)
    for gene in mutationGeneIndexs:
        # 确定变异基因位于第几个染色体
        chromosomeIndex = gene // n_gene
        # 确定变异基因位于当前染色体的第几个基因位
        geneIndex = gene % n_gene
        a = (np.random.random() * 2 - 1) + (np.random.random() * 2 - 1) * 1j
        update_population[chromosomeIndex, geneIndex] = population[chromosomeIndex, geneIndex] + a * 0.05
    return update_population


# def ws(pop):
#     pop_r = np.real(pop)
#     pop_i = np.imag(pop)
#     pop_r = np.max(np.min(pop_r, axis=1), -1)
#     pop_i = np.max(np.min(pop_i, axis=1), -1)
#     pop = np.complex(pop_r, pop_i)


def GA(max_iter=500):
    """
    10个种群繁衍500次，新种群为上次迭代选择交叉变异后的种群
    :param max_iter:
    :return:
    """
    # 每次迭代得到的最优解
    chromlength = 8
    optimalSolutions = []
    optimalValues = []
    # 决策变量的取值范围
    # decisionVariables = [[-3.0, 12.1], [4.1, 5.8]]
    population_size = 1000
    # 得到初始种群编码
    pop = geneEncoding(population_size, chromlength)
    for iteration in range(max_iter):
        # 得到个体适应度值和个体的累积概率
        fitnessvalues, cum_proba = getFitnessValue(pop)
        # 选择新的种群
        fitnessvalues, newpopulations = selectNewPopulation(pop, cum_proba, fitnessvalues)
        # 进行交叉操作
        crossoverpopulation = crossover(newpopulations)
        # mutation
        mutationpopulation = mutation(crossoverpopulation, chromlength)
        # update pop
        # pop = ws(pop)
        pop = mutationpopulation

        # 适应度评价
        # fitnessvalues, cum_individual_proba = getFitnessValue(fitnessFunction(), mutationpopulation)
        # 搜索每次迭代的最优解，以及最优解对应的目标函数的取值
        optimalValues.append(np.max(list(fitnessvalues)))  # 取适应度最大的值
        index = np.where(fitnessvalues == max(list(fitnessvalues)))
        optimalSolutions.append(mutationpopulation[index[0][0], :])  # 取适应度最大值对应的个体
    # 搜索最优解
    optimalValue = np.max(optimalValues)
    optimalIndex = np.where(optimalValues == optimalValue)
    optimalSolution = optimalSolutions[optimalIndex[0][0]]
    return optimalSolution, optimalValue


if __name__ == '__main__':
    import timeit

    solution, value = GA()
    print('最优解:')
    print(solution)
    print('最优目标函数值:', value)
    # 测量运行时间
    elapsedtime = timeit.timeit(stmt=GA, number=1)
    print('Searching Time Elapsed:(S)', elapsedtime)
