'''
全局特征提取
这里主要是想要提取噪底，判断噪底的功率大致是什么水平
不过如果可以的话也可以尝试提取信号区域，看看信号区域的功率大致是什么水平
'''
from statistics import median
import joblib
from calculate_NLL import calculate_NLL, fit_noise_gmm


'''
我想要做这样的归一化，就是如果一块区域完全是噪声，那么它的评分应该接近0，
如果一块区域完全是信号，那么它的评分应该接近1，
也就是说，假设一块噪底低功率区域的nll为3，一块信号高功率区域的nll为7，
我们需要把它分别映射到逼近0和逼近1的地方。

要做到这一点，我们需要知道噪声和信号的一般功率大致处于什么水平，
或者说噪声区域和信号区域的nll大致处于什么水平，
一开始，我想把整个频谱矩阵展平，取最小值作为噪声的水平，最大值作为信号的水平得了，
但后来发现这样做不行，原因很简单：频谱上的噪声可能大致处于-10dbm，信号的功率可能在10dbm左右，
但是整个频谱的所有点的功率最大值可能高达50dbm，最小值可能低至-50dbm，
所以我们不能仅仅把最小值或者最大值作为噪声或者信号的参考，
或许我们可以对所有点的功率值进行排序？然后取前10%的分位点作为信号的参考，后10%作为噪声的参考？
这样可能也行不通，因为如果频谱上的信号迹线极其稀疏的话，取前10%的分位点作为信号的参考显然是不合理的。

所以我现在有了一个新的思路，我们可以先找到功率最低的500个点，在里面随机取10个点，
然后以这10个点为中心，以比较小的w和h作为宽和高，相当于生成10个锚框，
这样的话这些锚框里大概率都是噪底区域，然后计算这十个锚框的nll，并取平均，以此作为噪底的nll水平参考。
同样的，我们可以先找到功率最高的500个点，在里面随机取10个点，以此生成信号的nll水平参考，
这样做的目的主要在于尽可能避免了孤立的离群点的影响，
'''

import pandas as pd
import random
import numpy as np

def generate_noise_anchors(df_origin, num_anchors=10,top_k=50,anchor_size=10):
    """
    生成噪声锚框，避免锚框重叠。
    num_anchors: 锚框的数量。
    top_k: 每轮从剩余高能量点中选前 top_k 作为候选
    返回值：一个包含锚框的列表，每个锚框为 [center_freq, center_time, width, height]。
    """
    #print("生成噪声锚框")
    anchors_list = []

    df = df_origin.copy()
    df_height, df_width = df.shape
    background_noise = df.values.min()

    # 初始化边界不采样
    df.iloc[0, :] = background_noise
    df.iloc[-1, :] = background_noise
    df.iloc[:, 0] = background_noise
    df.iloc[:, -1] = background_noise

    used_mask = np.zeros_like(df.values, dtype=bool)

    for _ in range(num_anchors):
        # 屏蔽掉已被覆盖区域的点
        masked_df = df.mask(used_mask, other=background_noise)

        # 提取高能量点
        flat_indices = np.argsort(masked_df.values.ravel())[:top_k]
        rows, cols = np.unravel_index(flat_indices, df.shape)

        if len(rows) == 0:
            print("高能量候选点不足，提前结束锚框生成。")
            break

        # 随机选一个点作为中心
        idx = np.random.choice(len(rows))
        y, x = rows[idx], cols[idx]

        w = anchor_size
        h = anchor_size

        # 截断尺寸防止越界
        w = min(w, df_width - x, x)
        h = min(h, df_height - y, y)

        anchors_list.append([x, y, w, h])

        # 更新 used_mask，将当前锚框区域标为 True
        x_min = max(0, x - w // 2)
        x_max = min(df_width, x + w // 2)
        y_min = max(0, y - h // 2)
        y_max = min(df_height, y + h // 2)
        used_mask[y_min:y_max, x_min:x_max] = True

    return anchors_list

def generate_signal_anchors(df_origin, num_anchors=10,top_k=50,anchor_size=10):
    """
        生成信号锚框，避免锚框重叠。
        num_anchors: 锚框的数量。
        top_k: 每轮从剩余高能量点中选前 top_k 作为候选
        返回值：一个包含锚框的列表，每个锚框为 [center_freq, center_time, width, height]。
        """
    #print("生成信号锚框")
    anchors_list = []

    df = df_origin.copy()
    df_height, df_width = df.shape
    background_noise = df.values.min()

    # 初始化边界不采样
    df.iloc[0, :] = background_noise
    df.iloc[-1, :] = background_noise
    df.iloc[:, 0] = background_noise
    df.iloc[:, -1] = background_noise

    used_mask = np.zeros_like(df.values, dtype=bool)

    for _ in range(num_anchors):
        # 屏蔽掉已被覆盖区域的点
        masked_df = df.mask(used_mask, other=background_noise)

        # 提取高能量点
        flat_indices = np.argsort(masked_df.values.ravel())[-top_k:]
        rows, cols = np.unravel_index(flat_indices, df.shape)

        if len(rows) == 0:
            print("高能量候选点不足，提前结束锚框生成。")
            break

        # 随机选一个点作为中心
        idx = np.random.choice(len(rows))
        y, x = rows[idx], cols[idx]


        w = anchor_size
        h = anchor_size

        # 截断尺寸防止越界
        w = min(w, df_width - x, x)
        h = min(h, df_height - y, y)

        anchors_list.append([x, y, w, h])

        # 更新 used_mask，将当前锚框区域标为 True
        x_min = max(0, x - w // 2)
        x_max = min(df_width, x + w // 2)
        y_min = max(0, y - h // 2)
        y_max = min(df_height, y + h // 2)
        used_mask[y_min:y_max, x_min:x_max] = True

    return anchors_list


def get_global_noise_nll(df_origin):
    '''计算噪底区域的nll'''
    gmm_model = fit_noise_gmm(df_origin, n_components=3, remove_top_percent=1.0)
    anchor_list = generate_noise_anchors(df_origin, num_anchors=50)

    nll_total=[]
    for anchor_box in anchor_list:
        nll=calculate_NLL(anchor_box, df_origin, gmm_model)
        nll_total.append(nll)

    print(f"平均噪声nll:{sum(nll_total)/len(nll_total)}")
    return sum(nll_total)/len(nll_total)

def get_global_signal_nll(df_origin):
    '''计算信号区域的nll'''
    gmm_model = fit_noise_gmm(df_origin, n_components=3, remove_top_percent=1.0)
    anchors_list = generate_signal_anchors(df_origin, num_anchors=50)

    nll_total = []
    for anchor_box in anchors_list:
        nll = calculate_NLL(anchor_box, df_origin, gmm_model)
        nll_total.append(nll)
    print(f"平均信号nll:{sum(nll_total) / len(nll_total)}")
    return sum(nll_total) / len(nll_total)


def sigmoid_for_nll(x,noise_nll,signal_nll):
    '''
    sigmoid函数，但不是原始的sigmoid
    主要作用是把正数压缩成0，1之间的小数
    '''
    median_nll=(noise_nll+signal_nll)/2
    z=10*(x-median_nll)/(signal_nll-noise_nll)
    return 1 / (1 + np.exp(-z))



if __name__=="__main__":
    df_origin = pd.read_csv(
        "dataset/train\DJI_phantom_4_pro_plus_5G_3.csv")
    df = df_origin.iloc[:, 1:]

    noise_nll=get_global_noise_nll(df)

    signal_nll=get_global_signal_nll(df)


    print(f"{sigmoid_for_nll(noise_nll,noise_nll,signal_nll)}")
    print(f"{sigmoid_for_nll(signal_nll, noise_nll, signal_nll)}")
    print(f"{sigmoid_for_nll(3, noise_nll, signal_nll)}")
    print(f"{sigmoid_for_nll(4, noise_nll, signal_nll)}")
    print(f"{sigmoid_for_nll(5, noise_nll, signal_nll)}")
    print(f"{sigmoid_for_nll(6, noise_nll, signal_nll)}")
    print(f"{sigmoid_for_nll(7, noise_nll, signal_nll)}")
    print(f"{sigmoid_for_nll(8, noise_nll, signal_nll)}")


