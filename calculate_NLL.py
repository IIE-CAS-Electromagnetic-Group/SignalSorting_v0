'''
用一维 GMM 对整个频谱做一次噪声建模，
再在每个锚框区域上用该模型计算平均负对数似然'''

import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
import joblib


def fit_noise_gmm(df_origin, n_components=3, remove_top_percent=1.0):
    """
    对整个频谱数据做一次噪声 GMM 建模，并返回已训练好的模型。

    参数:
      df_origin: pd.DataFrame 或 np.ndarray，shape = (H, W)，不含时间列
      n_components: GMM 分量数 K
      remove_top_percent: 先剔除最强信号的前 X% 能量值，再拟合

    返回:
      gmm: sklearn.mixture.GaussianMixture 对象
    """
    # 1. 展平成一维数组
    vals = df_origin.values.flatten() if hasattr(df_origin, 'values') else df_origin.flatten()

    # 2. 剔除前 remove_top_percent% 的最大值（避免信号污染噪声模型）
    if remove_top_percent > 0:
        thresh = np.percentile(vals, 100 - remove_top_percent)
        noise_vals = vals[vals <= thresh]
    else:
        noise_vals = vals

    # 3. 拟合 GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        reg_covar=1e-6,
        random_state=42
    )
    gmm.fit(noise_vals.reshape(-1, 1))
    return gmm


def calculate_NLL(anchor_box,df_origin,gmm):
    """
    基于已拟合的 GMM，对一个锚框区域计算评分（平均负对数似然）。

    参数:
      df_anchor: pd.DataFrame 或 np.ndarray，shape = (h, w)，区域内的能量值矩阵
      gmm: 已经拟合好的 sklearn.mixture.GaussianMixture 模型

    返回:
      score: float，区域的 anomaly score，越高说明越可能包含信号
    """
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

    arr_anchor = df_origin.values[t_min:t_max+1, f_min:f_max+1]

    # 1. 扁平化
    vals = arr_anchor.flatten()

    # 2. 计算每个点的 log-likelihood
    #    score_samples 返回 log p(x)，所以负 log-likelihood = - log p(x)
    logp = gmm.score_samples(vals.reshape(-1, 1))  # shape = (N,)
    nll = -logp

    # 3. 平均
    score = float(np.mean(nll))
    return score


# 示例用法
if __name__ == "__main__":


    # 假设 df_origin 是整个频谱的 DataFrame（不含时间列）
    df_origin = pd.read_csv("dataset/train/DJI_inspire_2_2G_0.csv").iloc[:, 1:]

    # 第一次只需建模一次
    gmm_model = fit_noise_gmm(df_origin, n_components=3, remove_top_percent=1.0)
    joblib.dump(gmm_model, "noise_gmm.pkl")#保存模型

    # 加载保存的模型
    gmm_model = joblib.load("noise_gmm.pkl")
    anchor_list=[[1250, 2150,80, 160]]
    score = calculate_NLL(anchor_list[0],df_origin, gmm_model)
    print(f"Anchor anomaly score: {score:.4f}")
