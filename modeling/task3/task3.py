import numpy as np
from GAP import GA
import matplotlib.pyplot as plt
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams.update({  # Use mathtext, not LaTeX
    'text.usetex': False,
    # Use the Computer modern font
    'font.serif': 'cmr10',
    'mathtext.fontset': 'cm',
})
solutions, _ = GA(1000)
snrs = np.linspace(0, 25, 26, dtype=int)
Iters = 100000
result = np.zeros((3, len(snrs)))

ampscale = math.sqrt(np.mean(abs(solutions) ** 2))


def qammod(bit1):
    """
    根据符号取出星座点
    :param bit1:
    :return:
    """
    sysmbol = [-1 + 1j, 1 + 1j, - 1 - 1j, 1 - 1j, - 2.731, 2.731 * 1j, - 2.731 * 1j, 2.731]
    return np.array(sysmbol)[bit1]


def qammod2_noise(bit1):
    """
    根据优化后的星座点并含噪声，与标准8QAM星座点位置，返回于之距离最近的符号索引
    :param bit1:
    :return:
    """
    out = list()
    sysmbol = [-1 + 1j, 1 + 1j, - 1 - 1j, 1 - 1j, - 2.731, 2.731 * 1j, - 2.731 * 1j, 2.731]
    for i in range(len(bit1)):
        dis = [abs(a - bit1[i]) for a in sysmbol]
        out.append(np.argmin(dis))
    return np.array(out)


def qammod3_noise(bit1, solutions):
    """
    根据优化后的星座点并含噪声，与标准8QAM星座点位置，返回于之距离最近的符号索引
    :param bit1:
    :param solutions:
    :return:
    """
    out = list()
    for i in range(len(bit1)):
        dis = [abs(a - bit1[i]) for a in solutions]
        out.append(np.argmin(dis))
    return np.array(out)


def dec2bin(bits):
    """
    十进制转二进制
    :param bits:
    :return:
    """
    data = list()
    for a in bits:
        b = bin(a).replace('0b', '')
        diff = 3 - len(b)
        while (diff):
            b = '0' + b
            diff -= 1
        data.append(b)
    return np.array(data)


for i in range(len(snrs)):  # 尝试添加多种信噪比

    # 仿真标准8QAM
    bit1 = np.random.randint(0, 8, int(Iters))
    sysm1 = qammod(bit1) / math.sqrt(4.7)

    # 仿真优化的8QAM
    bit2 = np.random.randint(0, 8, int(Iters))
    sysm2 = np.array(solutions[bit2]) / ampscale

    # 生成指定方差的噪声
    scale = (10 ** (-snrs[i] / 20)) / math.sqrt(2)
    noise_a = np.random.normal(scale=scale, size=int(Iters))
    noise_b = np.random.normal(scale=scale, size=int(Iters))
    noise = np.stack([np.complex(a, b) for a, b in zip(noise_a, noise_b)])

    noise_sysm1 = sysm1 + noise
    noise_sysm2 = sysm2 + noise

    noise_bit1 = qammod2_noise(noise_sysm1 * math.sqrt(4.7))
    noise_bit2 = qammod3_noise(noise_sysm2 * ampscale, solutions)

    # 测试信息熵
    fh = np.zeros((8, 1))
    for a in noise_bit1:
        fh[a] += 1
    fh = fh / Iters
    entropy = 0
    for elem in fh:
        entropy += elem * math.log(elem, 2)
    print(-entropy)
    Bc1 = dec2bin(noise_bit1) == dec2bin(bit1)
    Bc2 = dec2bin(noise_bit2) == dec2bin(bit2)

    result[1, i] = sum(Bc1 == 0) / Iters / 2
    result[2, i] = sum(Bc2 == 0) / Iters / 3
plt.plot(snrs, result[1], 'r', label='8QAM')
plt.plot(snrs, result[2], 'b', label='NewQAM')
ax = plt.gca()
ax.set_yscale('log')
plt.xlabel('SNR')
plt.ylabel('BER')
plt.title('BER与SNR关系曲线')
plt.legend()
plt.show()

Iters = 100
bit1 = np.random.randint(0, 8, int(Iters))
sysm1 = qammod(bit1) / math.sqrt(4.7)

bit2 = np.random.randint(0, 8, int(Iters))
sysm2 = np.array(solutions[bit2]) / ampscale
plt.subplot(211)
c = np.real(sysm1)
d = np.imag(sysm1)
for f, g in zip(c, d):
    plt.scatter(f, g)
plt.title('8QAM发射星座图')
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.xlabel('I')
plt.ylabel('Q')
plt.subplot(212)
c = np.real(sysm2)
d = np.imag(sysm2)
for f, g in zip(c, d):
    plt.scatter(f, g)
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.xlabel('I')
plt.ylabel('Q')
plt.title('newQAM发射星座图')
plt.tight_layout()
plt.show()

