import math
import pandas as pd
from Kruskal import Kruskal
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_transfer_rate(distance):
    """
    计算传输距离对应的传输速率
    :param distance:
    :return:
    """
    if distance == 0:
        return 0
    if distance <= 600:
        rate = 32
    elif distance <= 1200:
        rate = 16
    elif distance <= 3000:
        rate = 8
    else:
        rate = 0
    return rate


def cal_network_traffic(distance, pop1, pop2):
    """
    计算两地的网络价值，即权重
    :param pop1:
    :param pop2:
    :return: 
    """
    rate = get_transfer_rate(distance)
    return math.sqrt(pop1 * pop2) * rate


def plot_result(plot_data, title, total_value):
    data = pd.read_excel('task2_city_data.xls', )
    pos_dict = dict()
    for row in data.itertuples():
        pos_dict.setdefault(row[0], (float(row[3]), float(row[4])), )
    longitude = []
    latitude = []
    citys = []
    for item in list(plot_data):
        start = item[1]
        end = item[2]
        longitude.append([pos_dict[start][0], pos_dict[end][0]])
        latitude.append([pos_dict[start][1], pos_dict[end][1]])
        citys.append([start, end])
    # print(longitude)
    for long, lati, city in zip(longitude, latitude, citys):
        plt.text(long[0] + 0.5, lati[0], city[0])
        plt.text(long[1] + 0.5, lati[1], city[1])
        plt.plot(long, lati, 'r', marker='o')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('{0} {1:.3f}'.format(title, total_value))
    plt.show()


def main():
    num_city = 12
    populations = {'北京&天津': 37.3, '哈尔滨': 10.9, '西安': 9.6, '乌鲁木齐': 2.2, '郑州': 10.0, '拉萨': 0.9, '成都': 16.0, '重庆': 30.5,
                   '武汉': 10.9, '上海': 24.0, '昆明': 6.7, '广州&深圳': 32.6}
    # populations = {'北京&天津': 19.61, '哈尔滨': 10.63, '西安': 8.46, '乌鲁木齐': 3.11, '郑州': 8.62, '拉萨': 0.55, '成都': 14.04,
    #                '重庆': 28.84, '武汉': 9.78, '上海': 23.01, '昆明': 6.43, '广州&深圳': 10.35}
    city_list = list(populations.keys())
    # 读取任务数据
    data = pd.read_excel('task2_distance_data.xls', )
    distance_mat = list()
    for index, row in enumerate(data.itertuples()):
        distances = list()
        for i in range(num_city):
            distances.append(row[i + 2])
        distance_mat.append(distances)

    # 计算城市间的网络价值
    values_mat = list()
    for i, row in enumerate(distance_mat):
        values = list()
        for j, elem in enumerate(row):
            value = cal_network_traffic(elem, populations[city_list[i]], populations[city_list[j]])
            values.append(value)
        values_mat.append(values)

    edges = list()
    for i in range(num_city):
        for j in range(i, num_city):
            item = (values_mat[i][j], city_list[j], city_list[i])
            edges.append(item)

    # print(edges)
    graph = {'vertices': city_list,
             'edges': edges}

    result = Kruskal(graph).kruskal()
    print(result)
    # 对剩余的连接按网络价值递减排序
    remains = set(edges) - set(result)
    remains_5 = sorted(list(remains), reverse=True)[:5]
    total_value1 = 0
    for row in result:
        total_value1 += row[0]
    total_value = total_value1
    for row in remains_5:
        total_value += row[0]
    print(remains_5)
    print('连接数为16的网络价值', total_value)
    # print(type(result))
    plot_result(list(result) + remains_5, '连接数为16的网络规划的网络价值为', total_value)
    total_value = total_value1
    remains_22 = sorted(list(remains), reverse=True)[:22]
    for row in remains_22:
        total_value += row[0]
    print('连接数为33的网络价值', total_value)
    plot_result(list(result) + remains_22, '连接数为33的网络规划的网络价值为', total_value)


if __name__ == '__main__':
    main()
