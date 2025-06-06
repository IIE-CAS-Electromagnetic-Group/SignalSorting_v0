'''
计算奖励函数
奖励函数是给强化学习用的，相当于给它的这一动作进行评分，
奖励函数的设计需要鼓励模型快速地、准确地通过“动作”调整锚框使其框选到信号迹线，
'''
import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import torch
import joblib

from global_features_extract import sigmoid_for_nll
from utils.plot_greyscale import *
from calculate_NLL import calculate_NLL
from scipy.ndimage import label


def sigmoid(x):
    '''
    sigmoid函数，但不是原始的sigmoid
    主要作用是把正数压缩成0，1之间的小数
    '''
    return 1 / (1 + np.exp(-x + 5))


def calculate_reward(old_anchor, new_anchor, df_origin):
    '''
    新版本奖励函数
    奖励“新加进来的区域有用”，惩罚“丢掉有用区域”
    不盲目鼓励面积，只关注“是否真正增益了内容”
    '''
    old_score = calculate_score(old_anchor, df_origin)
    new_score = calculate_score(new_anchor, df_origin)
    delta_score = new_score - old_score

    old_area = old_anchor[2] * old_anchor[3]
    new_area = new_anchor[2] * new_anchor[3]

    marginal_gain = 0  # 新引入区域的平均能量 / 评分（只在扩大 / 平移时有）
    marginal_loss = 0  # 被丢弃区域的平均能量 / 评分（只在缩小/平移时有）

    old_freq_min = old_anchor[0] - old_anchor[2] // 2
    old_freq_min = max(old_freq_min, 0)
    old_freq_max = old_anchor[0] + old_anchor[2] // 2
    old_freq_max = min(old_freq_max, df_origin.shape[1] - 1)

    new_freq_min = new_anchor[0] - new_anchor[2] // 2
    new_freq_min = max(new_freq_min, 0)
    new_freq_max = new_anchor[0] + new_anchor[2] // 2
    new_freq_max = min(new_freq_max, df_origin.shape[1] - 1)

    old_time_min = old_anchor[1] - old_anchor[3] // 2
    old_time_min = max(old_time_min, 0)
    old_time_max = old_anchor[1] + old_anchor[3] // 2
    old_time_max = min(old_time_max, df_origin.shape[0] - 1)

    new_time_min = new_anchor[1] - new_anchor[3] // 2
    new_time_min = max(new_time_min, 0)
    new_time_max = new_anchor[1] + new_anchor[3] // 2
    new_time_max = min(new_time_max, df_origin.shape[0] - 1)

    if new_anchor[2] != old_anchor[2]:  # 说明宽度发生了变化，可能是宽度（频率）收缩或扩张

        if new_anchor[2] > old_anchor[2]:  # 扩张操作
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, new_freq_min:old_freq_min]) + calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, old_freq_max + 1:new_freq_max + 1])
        else:  # 收缩操作
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, old_freq_min:new_freq_min]) + calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, new_freq_max:old_freq_max + 1])
    elif new_anchor[3] != old_anchor[3]:  # 高度发生了变化，可能是高度（时间）收缩或扩张
        if new_anchor[3] > old_anchor[3]:  # 扩张操作
            marginal_gain += calculate_df_score(
                df_origin.iloc[new_time_min:old_time_min, old_freq_min:old_freq_max + 1]) + calculate_df_score(
                df_origin.iloc[old_time_max + 1:new_time_max + 1, old_freq_min:old_freq_max + 1])
        else:  # 收缩操作
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:new_time_min, old_freq_min:old_freq_max + 1]) + calculate_df_score(
                df_origin.iloc[new_time_max:old_time_max, old_freq_min:old_freq_max + 1])
    else:  # 如果不是在宽度或高度上收缩或扩张的话，应该就是平移了，这里需要同时考虑到失去的区域和得到的区域的评分
        if new_anchor[0] < old_anchor[0]:  # 频率向左边平移
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, new_freq_min:old_freq_min])
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, new_freq_max:old_freq_max + 1])
            pass
        elif new_anchor[0] > old_anchor[0]:  # 频率往右边平移了
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, old_freq_min:new_freq_min])
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, old_freq_max + 1:new_freq_max + 1])
            pass
        elif new_anchor[1] < old_anchor[1]:  # 时间往上平移了
            marginal_gain += calculate_df_score(
                df_origin.iloc[new_time_min:old_time_min, old_freq_min:old_freq_max + 1])
            marginal_loss += calculate_df_score(
                df_origin.iloc[new_time_max:old_time_max, old_freq_min:old_freq_max + 1])
            pass
        elif new_anchor[1] > old_anchor[1]:  # 时间往下平移了
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:new_time_min, old_freq_min:old_freq_max + 1])
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_max + 1:new_time_max + 1, old_freq_min:old_freq_max + 1])
    # 最终奖励

    '''
    marginal_gain-marginal_loss本质上就是锚框调整时的框进来的新区域的评分和被移出去的区域的评分之差。
    这一项的目的主要是希望锚框调整过程中能尽可能把功率高的区域框进来，同时尽可能避免把功率高的区域丢出去。
    比如说吧，锚框往左平移了，那么左侧移进来的区域评分是否高呢？右侧被移出去的评分是否比较低呢？
    （不过评分的范围是[0,1]，没有负数）
    '''

    reward = delta_score + 0.008 * (marginal_gain - marginal_loss)

    return reward


def calculate_reward_new(old_anchor, new_anchor, df_origin, weight_list):
    """
    新版本奖励函数（带权重分量输出）
    :param old_anchor: 原始锚框 (x, y, w, h)
    :param new_anchor: 调整后的锚框 (x, y, w, h)
    :param df_origin: 原始信号数据
    :param weight_list: 当前 reward softmax 权重（用于监督）
    :return: reward 值，reward 分量 [delta_score, gain_minus_loss]
    """


    old_score = calculate_score(old_anchor, df_origin)
    new_score = calculate_score(new_anchor, df_origin)
    delta_score = new_score - old_score

    area_gain = calculate_area_gain(old_anchor, new_anchor, df_origin)
    boundary_contrast_gain = compute_boundary_contrast(new_anchor, df_origin) - compute_boundary_contrast(old_anchor,
                                                                                                          df_origin)
    connectivity_loss = compute_connectivity_loss(new_anchor, df_origin, alpha=1.0, min_ratio=0.1)
    width_rate_loss = compute_width_rate_loss(new_anchor, max_ratio=6.0)
    height_rate_loss = compute_height_rate_loss(new_anchor, max_ratio=6.0)

    # === Apply static scaling factors to each reward component ===
    delta_score *= 100
    area_gain *= 3.0
    boundary_contrast_gain *= 10
    connectivity_loss *= 2
    width_rate_loss *= 0.5
    height_rate_loss *= 0.5

    reward = weight_list[0] * delta_score + weight_list[1] * area_gain + weight_list[2] * boundary_contrast_gain - \
             weight_list[3] * connectivity_loss - weight_list[4] * width_rate_loss - weight_list[5] * height_rate_loss

    # delta_score: 目标函数主项，表示新锚框的平均强度提升（大致是0.1左右吧）
    # area_gain: 鼓励把高能量区域框进来，惩罚丢弃强信号（最大能到1）
    # boundary_contrast: 鼓励锚框边缘接近能量差异大的区域（靠近信号边缘）（最大0.25）
    # connectivity_loss: 惩罚锚框包含多个独立信号（说明覆盖太多，或没有精细覆盖）
    # width_rate_loss: 惩罚锚框过宽（防止扁平）
    # height_rate_loss: 惩罚锚框过高（防止细长）





    reward = math.tanh(reward)


    #print(f"all:{reward} delta_score:{delta_score},area_gain:{area_gain},boundary_contrast_gain:{boundary_contrast_gain},connectivity_loss:{connectivity_loss},width_rate_loss:{width_rate_loss},height_rate_loss:{height_rate_loss}")
    return reward, [delta_score, area_gain,boundary_contrast_gain,connectivity_loss,width_rate_loss,height_rate_loss]


# -----------------------------------------------------


def calculate_score(anchor_box, df_origin):
    # 提取锚框区域
    cf, ct, w, h = anchor_box
    f_min = int(cf - w / 2)
    f_max = int(cf + w / 2)
    t_min = int(ct - h / 2)
    t_max = int(ct + h / 2)

    # 边界检查
    f_min = max(0, f_min)
    f_max = min(df_origin.shape[1], f_max)
    t_min = max(0, t_min)
    t_max = min(df_origin.shape[0], t_max)
    # 提取区域并转换为NumPy数组
    region_inner = df_origin.iloc[t_min:t_max + 1, f_min:f_max + 1].values
    # 计算统计量
    mean_inner = np.mean(region_inner) if region_inner.size > 0 else 0.0
    # 评分项计算
    score1 = mean_inner
    # 最终评分
    final_score = score1
    return float(final_score)


def calculate_area_gain(old_anchor, new_anchor, df_origin):
    '''
    计算扩展区域带来的新增功率平均值
    marginal_gain-marginal_loss本质上就是锚框调整时的框进来的新区域的评分和被移出去的区域的评分之差。
    这一项的目的主要是希望锚框调整过程中能尽可能把功率高的区域框进来，同时尽可能避免把功率高的区域丢出去。
    比如说吧，锚框往左平移了，那么左侧移进来的区域评分是否高呢？右侧被移出去的评分是否比较低呢？
    （不过评分的范围是[0,1]，没有负数）
    '''
    old_freq_min = max(old_anchor[0] - old_anchor[2] // 2, 0)
    old_freq_max = min(old_anchor[0] + old_anchor[2] // 2, df_origin.shape[1] - 1)
    new_freq_min = max(new_anchor[0] - new_anchor[2] // 2, 0)
    new_freq_max = min(new_anchor[0] + new_anchor[2] // 2, df_origin.shape[1] - 1)

    old_time_min = max(old_anchor[1] - old_anchor[3] // 2, 0)
    old_time_max = min(old_anchor[1] + old_anchor[3] // 2, df_origin.shape[0] - 1)
    new_time_min = max(new_anchor[1] - new_anchor[3] // 2, 0)
    new_time_max = min(new_anchor[1] + new_anchor[3] // 2, df_origin.shape[0] - 1)

    marginal_gain = 0.0
    marginal_loss = 0.0

    if new_anchor[2] != old_anchor[2]:# 说明宽度发生了变化，可能是宽度（频率）收缩或扩张
        if new_anchor[2] > old_anchor[2]:
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, new_freq_min:old_freq_min])
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, old_freq_max + 1:new_freq_max + 1])
            marginal_gain=marginal_gain-1#这里的减法是为了确定扩张进来的面积属于信号区域，鼓励扩进来平均大于0.5的区域，不鼓励扩进来平均小于0.5的区域
        else:
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, old_freq_min:new_freq_min])
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, new_freq_max:old_freq_max + 1])
            marginal_loss = marginal_loss - 1
    elif new_anchor[3] != old_anchor[3]:# 高度发生了变化，可能是高度（时间）收缩或扩张
        if new_anchor[3] > old_anchor[3]:
            marginal_gain += calculate_df_score(
                df_origin.iloc[new_time_min:old_time_min, old_freq_min:old_freq_max + 1])
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_max + 1:new_time_max + 1, old_freq_min:old_freq_max + 1])
            marginal_gain = marginal_gain - 1
        else:
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:new_time_min, old_freq_min:old_freq_max + 1])
            marginal_loss += calculate_df_score(
                df_origin.iloc[new_time_max:old_time_max, old_freq_min:old_freq_max + 1])
            marginal_loss = marginal_loss - 1
    else:# 如果不是在宽度或高度上收缩或扩张的话，应该就是平移了，这里需要同时考虑到失去的区域和得到的区域的评分
        if new_anchor[0] < old_anchor[0]:# 频率向左边平移
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, new_freq_min:old_freq_min])
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, new_freq_max:old_freq_max + 1])
            marginal_gain = marginal_gain - 0.5
            marginal_loss = marginal_loss - 0.5
        elif new_anchor[0] > old_anchor[0]:# 频率向右边平移
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, old_freq_min:new_freq_min])
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_min:old_time_max + 1, old_freq_max + 1:new_freq_max + 1])
            marginal_gain = marginal_gain - 0.5
            marginal_loss = marginal_loss - 0.5
        elif new_anchor[1] < old_anchor[1]:# 时间往上平移了
            marginal_gain += calculate_df_score(
                df_origin.iloc[new_time_min:old_time_min, old_freq_min:old_freq_max + 1])
            marginal_loss += calculate_df_score(
                df_origin.iloc[new_time_max:old_time_max, old_freq_min:old_freq_max + 1])
            marginal_gain = marginal_gain - 0.5
            marginal_loss = marginal_loss - 0.5
        elif new_anchor[1] > old_anchor[1]:# 时间往下平移了
            marginal_loss += calculate_df_score(
                df_origin.iloc[old_time_min:new_time_min, old_freq_min:old_freq_max + 1])
            marginal_gain += calculate_df_score(
                df_origin.iloc[old_time_max + 1:new_time_max + 1, old_freq_min:old_freq_max + 1])
            marginal_gain = marginal_gain - 0.5
            marginal_loss = marginal_loss - 0.5

    area_gain = marginal_gain - marginal_loss
    '''这个函数也有其特点：它整体上其实是鼓励锚框扩张了，反对锚框收缩的
    因为噪声区域的评分其实也不低于0（尽管趋近于0），所以如果收缩区域（完全没有任何新扩进来的区域，marginal_gain=0）了，即使收缩的是噪声区域，也会使得area_gain=(0-marginal_loss)<0
    '''
    return area_gain


def compute_boundary_contrast(anchor_box, df_origin):
    '''
    计算单个锚框边缘与外部对比度

    基于锚框边缘区域方差计算边界对比度

    这个自然是对比度越高越好，鼓励贴边扩张
    锚框不能缩在信号区域内部，否则对比度会很低
    锚框也不能往外边疯狂扩张，否则把大量噪声信号吸纳进来，对比度也会很低。
    这里我们计算对比度是通过直接提取锚框边缘的一个空心矩阵计算的，这个空心矩阵处于锚框边缘的区域，计算它的方差，方差越大说明这个边缘对比度越高
    '''
    # 提取锚框区域
    cf, ct, w, h = anchor_box
    f_min = int(cf - w / 2)
    f_max = int(cf + w / 2)
    t_min = int(ct - h / 2)
    t_max = int(ct + h / 2)

    # 边界检查
    f_min = max(0, f_min)
    f_max = min(df_origin.shape[1], f_max)
    t_min = max(0, t_min)
    t_max = min(df_origin.shape[0], t_max)

    freq_thickness = max(1, int((f_max - f_min) * 0.2))
    time_thickness = max(1, int((t_max - t_min) * 0.2))

    region_edge = df_origin.iloc[max(0, t_min - time_thickness):t_min + time_thickness,
                  f_min:f_max + 1].values.flatten()
    region_edge1 = df_origin.iloc[t_max - time_thickness:min(t_max + time_thickness, df_origin.shape[0]),
                   f_min:f_max + 1].values.flatten()
    region_edge = np.concatenate((region_edge, region_edge1), axis=0)
    region_edge1 = df_origin.iloc[t_min:t_max + 1,
                   max(0, f_min - freq_thickness):f_min + freq_thickness].values.flatten()
    region_edge = np.concatenate((region_edge, region_edge1), axis=0)
    region_edge1 = df_origin.iloc[t_min:t_max + 1,
                   f_max - freq_thickness:min(f_max + freq_thickness, df_origin.shape[1])].values.flatten()
    region_edge = np.concatenate((region_edge, region_edge1), axis=0)
    return float(np.var(region_edge))


def compute_connectivity_loss(anchor, df, alpha=1.0, min_ratio=0.01):
    '''计算信号联通性惩罚
    为什么是惩罚而不是奖励呢？
    因为我们需要鼓励一个锚框里面只有一个信号迹线，而不能同时框进来多个信号迹线

    我们并不知道如何确定“高能量域”的能量阈值，
    这个高能量域的判定是不是应该给一个最小区域范围限制呢？
    它会不会把噪声点也当成高能量域呢？
    会不会把一个高能量域当成多个高能量域呢？
    '''
    # Step 1: 计算全局能量阈值（从整张图中统计）
    '''global_mean = df.values.mean()
    global_std = df.values.std()
    threshold = global_mean + alpha * global_std'''

    threshold=0.8#这里干脆直接手动设置一个得了

    # Step 2: 截取锚框对应的子区域
    cf, ct, w, h = map(int, anchor)
    f_min = max(cf - w // 2, 0)
    f_max = min(cf + w // 2, df.shape[1])
    t_min = max(ct - h // 2, 0)
    t_max = min(ct + h // 2, df.shape[0])
    region = df.iloc[t_min:t_max, f_min:f_max].values

    # Step 3: 构造高能量二值图（统一使用全局阈值）
    binary = (region > threshold).astype(int)

    # Step 4: 连通域标记
    labeled, num_features = label(binary)

    # Step 5: 统计连通区域面积，筛除面积太小的伪迹线
    areas = [(labeled == i).sum() for i in range(1, num_features + 1)]
    area_threshold = min_ratio * region.size
    large_areas = [a for a in areas if a >= area_threshold]

    # Step 6: 只允许有一个主 signal trace，其余都惩罚
    penalty = max(0, len(large_areas) - 1)
    return float(penalty)


def compute_width_rate_loss(anchor, max_ratio=3.0):
    """
    锚框宽高比过大（太扁平）时给予惩罚。

    参数：
    - anchor: [cf, ct, w, h]
    - max_ratio: 允许的最大宽高比，超过这个就惩罚

    返回：
    - 惩罚值（大于等于 0）
    """
    _, _, w, h = map(float, anchor)
    if h == 0: return 1e6  # 防止除0
    width_ratio = w / h
    return max(0.0, width_ratio - max_ratio)


def compute_height_rate_loss(anchor, max_ratio=3.0):
    """
    锚框宽高比过大（太扁平）时给予惩罚。

    参数：
    - anchor: [cf, ct, w, h]
    - max_ratio: 允许的最大宽高比，超过这个就惩罚

    返回：
    - 惩罚值（大于等于 0）
    """
    _, _, w, h = map(float, anchor)
    if w == 0: return 1e6  # 防止除0
    height_ratio = h / w
    return max(0.0, height_ratio - max_ratio)


def calculate_df_score(df):
    '''计算df区域内的评分（不需要传入锚框了）'''
    if df.shape[0] == 0 or df.shape[1] == 0:
        return 0.0

    region_inner = df.values
    mean_inner = np.mean(region_inner) if region_inner.size > 0 else 0.0
    return mean_inner

def calculate_average_score(anchor_list,df_origin):
    '''
    计算锚框评分的整体平均值
    '''
    total_score=0
    for anchor in anchor_list:
        total_score+=calculate_score(anchor, df_origin)
    return total_score/len(anchor_list)

def calculate_score_by_nll(anchor_box, df_origin, gmm_model, noise_nll, signal_nll):
    '''
    通过NLL计算锚框评分
    这里需要提供噪声背景的NLL水平和信号区域的NLL水平，以便于将锚框内的NLL映射到0与1之间。
    '''
    NLL = calculate_NLL(anchor_box, df_origin, gmm_model)
    return sigmoid_for_nll(NLL, noise_nll, signal_nll)


if __name__ == "__main__":
    # 使用示例
    '''df_origin=pd.read_csv("D:\\20250310迹线预处理3_30M\\111\\0_9000MHz原始信号数据\\20250310122051_20250310125051_1800.0072_2700.0063.csv")
    df = df_origin.iloc[:1000, 1:1500]
    print("estimate_background_noise:"+str(estimate_background_noise(df)))
    print("estimate_signal_energy:" + str(estimate_signal_energy(df,2)))
    anchor_box = [1000, 107, 50, 48]
    #plot_greyscale_for_singledf(df, "g0.png")
    #plot_greyscale_for_singledf_with_anthor(df, [anchor_box], "g0_anchor.png")
    print("分数0：" + str(calculate_score(anchor_box, df)))

    anchor_box1 = [1045, 107, 50, 48]
    #plot_greyscale_for_singledf(df, "g1.png")
    #plot_greyscale_for_singledf_with_anthor(df, [anchor_box], "g1_anchor.png")
    print("分数1：" + str(calculate_score(anchor_box1, df)))



    anchor_box2 = [1060, 107, 50, 48]
    #plot_greyscale_for_singledf(df,"g2.png")
    #plot_greyscale_for_singledf_with_anthor(df,[anchor_box],"g3_anchor.png")
    print("分数2：" + str(calculate_score(anchor_box2, df)))


    reward=calculate_reward(anchor_box1, anchor_box2, df)
    print(f"综合奖励值: {reward:.2f}")

    reward = calculate_reward(anchor_box2, anchor_box1, df)
    print(f"综合奖励值: {reward:.2f}")

    anchor_box = [1066, 107, 62, 48]
    #plot_greyscale_for_singledf_with_anthor(df, [anchor_box], "g4_anchor.png")
    print("分数3：" + str(calculate_score(anchor_box, df)))

    anchor_box = [1066, 107, 62, 214]
    #plot_greyscale_for_singledf_with_anthor(df, [anchor_box], "g5_anchor.png")
    print("分数4：" + str(calculate_score(anchor_box, df)))

    anchor_box = [1066, 500, 62, 999]
    #plot_greyscale_for_singledf_with_anthor(df, [anchor_box], "g6_anchor.png")
    print("分数5：" + str(calculate_score(anchor_box, df)))'''

    pass
