'''数据集预处理
原始频谱 df 是功率谱密度（PSD），比如单位是 dB（对数能量值），
常见数值范围可能在 [-120, -20] 之间，很多点是负值且幅度跨度大。

这种原始数据直接送入CNN去做特征提取有几个问题：
数值跨度大（比如 [-120, -20]）——导致梯度更新不稳定，优化困难
分布偏斜（噪声密集区 vs 信号稀疏区）——模型容易偏向低能量区域，忽略稀有信号模式
负值存在——对 ReLU（只保留正值）不友好，易造成特征损失
'''
import pandas as pd
from global_features_extract import generate_signal_anchors
import numpy as np

def df_normalization(df_origin):
    '''将df归一化，线性映射到 [0, 1]'''
    # 确定底噪
    all_values = df_origin.values.flatten()
    background_noise = pd.Series(all_values).median()

    signal_max = df_origin.values.max()
    if signal_max>background_noise+20:
        signal_max=background_noise+20

    powerwidth = signal_max - background_noise

    # 使用 Pandas 的向量化操作来设置底噪
    df_clipped = df_origin.clip(lower=background_noise,upper=signal_max)

    df_new = (df_clipped - background_noise) / powerwidth
    return df_new

def df_normalization_nonlinear(df_origin):
    '''将df归一化，非线性映射到 [0, 1]
    这里需要借助一个类似sigmoid函数的玩意'''
    # 确定底噪
    all_values = df_origin.values.flatten()
    background_noise = pd.Series(all_values).median()

    '''信号大致处于啥水平呢，这里采用之前那个随机采样信号锚框的操作估计'''
    signal_anchors=generate_signal_anchors(df_origin, num_anchors=10, top_k=50, anchor_size=10)
    all_signal_power=0
    for signal_anchor in signal_anchors:
        # 提取锚框区域
        cf, ct, w, h = signal_anchor
        f_min = int(cf - w / 2)
        f_max = int(cf + w / 2)
        t_min = int(ct - h / 2)
        t_max = int(ct + h / 2)

        # 边界检查
        f_min = max(0, f_min)
        f_max = min(df_origin.shape[1], f_max)
        t_min = max(0, t_min)
        t_max = min(df_origin.shape[0], t_max)
        arr_anchor = df_origin.values[t_min:t_max + 1, f_min:f_max + 1]
        all_signal_power+=np.mean(arr_anchor)

    signal_max=all_signal_power/len(signal_anchors)

    powerwidth = signal_max - background_noise

    #先对df_origin进行一个压缩映射，其中background_noise映射到-5上，signal_max的值映射到+5上
    df_tmp=10*(df_origin-background_noise)/powerwidth+(-5)

    #然后非线性映射到(0,1)区间上
    df_new = 1/(1+np.exp(-df_tmp))
    return df_new
