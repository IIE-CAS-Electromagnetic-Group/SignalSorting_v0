'''锚框初始化'''
from utils.plot_greyscale import *
import pandas as pd
import random
import numpy as np


def generate_initial_anchors_old_v0(df_origin,num_anchors=50):
    '''
    生成初始锚框
    num_anchors是初始锚框的数量，
    返回的是一个列表
    :return:
    '''
    #print("...生成初始锚框...")
    anchors_list=[]
    # 确定底噪
    all_values = df_origin.values.flatten()
    background_noise = pd.Series(all_values).median()
    #print(f"底噪{background_noise}")


    df_width = df_origin.shape[1]  # 列数对应宽
    df_height = df_origin.shape[0]  # 行数对应高


    # 找到信号较高的点，这里假设我们取前 1000 个较高的点作为候选
    df_tmp=df_origin
    df_tmp.iloc[0,:]=background_noise
    df_tmp.iloc[-1, :] = background_noise
    df_tmp.iloc[:, 0] = background_noise
    df_tmp.iloc[:, -1] = background_noise
    #print(df_tmp)

    flat_indices = np.argsort(df_tmp.values.ravel())[-(num_anchors*100):]  # 取前 500 个较高的点的展平索引
    rows, cols = np.unravel_index(flat_indices, df_tmp.shape)  # 转换为二维索引

    # 随机选取十个点
    selected_indices = np.random.choice(len(rows), num_anchors, replace=False)

    # 获取选中的行和列索引
    selected_rows = rows[selected_indices]
    selected_cols = cols[selected_indices]
    for i in range(num_anchors):
        w = max(10,min(30,int(random.randint(min(5,min(selected_cols[i],df_width-selected_cols[i])*2), max(5,min(selected_cols[i],df_width-selected_cols[i])*2))*0.2)))
        h = int(w * 0.2)
        w_safe=min(df_origin.shape[1]-selected_cols[i],selected_cols[i])
        h_safe=min(df_origin.shape[0]-selected_rows[i],selected_rows[i])
        w=min(w,w_safe)
        h=min(h,h_safe)

        anchors_list.append([selected_cols[i], selected_rows[i],w, h])

    return anchors_list


def calculate_iou(anchor1, anchor2):
    """
    计算两个锚框的交并比（IoU）
    :param anchor1: [center_freq, center_time, width, height]
    :param anchor2: [center_freq, center_time, width, height]
    :return: IoU值
    """
    x1_min = anchor1[0] - anchor1[2] / 2
    x1_max = anchor1[0] + anchor1[2] / 2
    y1_min = anchor1[1] - anchor1[3] / 2
    y1_max = anchor1[1] + anchor1[3] / 2

    x2_min = anchor2[0] - anchor2[2] / 2
    x2_max = anchor2[0] + anchor2[2] / 2
    y2_min = anchor2[1] - anchor2[3] / 2
    y2_max = anchor2[1] + anchor2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def merge_anchor(anchor_list, iou_threshold=0.1):
    """
    合并重叠锚框
    :param anchor_list: 锚框列表，每个锚框格式为 [center_freq, center_time, width, height]
    :param iou_threshold: IoU阈值，用于判断是否需要合并
    :return: 合并后的锚框列表
    """
    if not anchor_list:
        return []

    # 按宽度排序，便于后续处理
    anchor_list = sorted(anchor_list, key=lambda x: x[2], reverse=True)

    merged_anchors = []
    while anchor_list:
        base_anchor = anchor_list.pop(0)
        to_merge = [base_anchor]

        # 检查和其他锚框的IoU
        rest_anchors = []
        for anchor in anchor_list:
            iou_list=[]#anchor要与每一个to_merge列表里的锚框比较，而不仅仅跟第一个base_anchor比较
            for merged_anchor in to_merge:
                iou = calculate_iou(merged_anchor, anchor)
                iou_list.append(iou)
            if max(iou_list) > iou_threshold:
                to_merge.append(anchor)
            else:
                rest_anchors.append(anchor)

        # 合并所有需要合并的锚框
        if len(to_merge) > 1:
            x_min = min([a[0] - a[2] / 2 for a in to_merge])
            x_max = max([a[0] + a[2] / 2 for a in to_merge])
            y_min = min([a[1] - a[3] / 2 for a in to_merge])
            y_max = max([a[1] + a[3] / 2 for a in to_merge])

            merged_anchor = [
                (x_min + x_max) / 2,  # center_freq
                (y_min + y_max) / 2,  # center_time
                x_max - x_min,        # width
                y_max - y_min         # height
            ]
            merged_anchors.append(merged_anchor)
        else:
            merged_anchors.append(base_anchor)

        anchor_list = rest_anchors

    return merged_anchors





def generate_initial_anchors_old_v1(df_origin, num_anchors=50, max_retries=100):
    """
    生成初始锚框，避免锚框重叠。
    num_anchors: 初始锚框的数量。
    max_retries: 每个锚框生成的最大重试次数，用于防止死循环。
    返回值：一个包含锚框的列表，每个锚框为 [center_freq, center_time, width, height]。

    这个函数已经弃用了，它的缺点是：
    选点不更新:所有候选中心点是一次性随机采样，即使被用过或落入已有锚框区域，也不会动态更新候选池
    高能区集中时失败率高:如果高能区域集中，一旦占用前几个锚框，后续高能点很容易全部“撞锚”，触发 retries 超限
    采样效率低:会在已知无效区域重复尝试，浪费计算时间
    """
    print("生成初始锚框...")
    anchors_list = []
    all_values = df_origin.values.flatten()
    background_noise = pd.Series(all_values).median()
    signal_max = df_origin.values.max()

    df_width = df_origin.shape[1]  # 列数对应宽
    df_height = df_origin.shape[0]  # 行数对应高

    df_tmp = df_origin.copy()
    df_tmp.iloc[0, :] = background_noise
    df_tmp.iloc[-1, :] = background_noise
    df_tmp.iloc[:, 0] = background_noise
    df_tmp.iloc[:, -1] = background_noise

    flat_indices = np.argsort(df_tmp.values.ravel())[-(num_anchors * 100):]
    rows, cols = np.unravel_index(flat_indices, df_tmp.shape)
    selected_indices = np.random.choice(len(rows), num_anchors * 5, replace=False)
    selected_rows = rows[selected_indices]
    selected_cols = cols[selected_indices]

    def is_center_inside_existing_anchors(x, y, anchors):
        """检查中心点是否落在已有锚框内"""
        for anchor in anchors:
            cx, cy, w, h = anchor
            if abs(cx - x) <= w / 2 and abs(cy - y) <= h / 2:
                return True
        return False

    retries = 0
    for i in range(len(selected_rows)):
        if len(anchors_list) >= num_anchors:
            break

        x = selected_cols[i]
        y = selected_rows[i]

        if is_center_inside_existing_anchors(x, y, anchors_list):
            retries += 1
            if retries >= max_retries:
                print("已达到最大重试次数，跳过当前点")
                continue
            else:
                continue  # 继续尝试下一个点


        w_max=min(x, df_width - x)*2#w可以取的最大值（不能使锚框出界）
        w=random.randint(0,int(w_max*0.1))
        if w<15:
            w=15#w不能小于这个值

        h_max=min(y,df_height-y)*2#h可以取的最大值
        h=random.randint(0,int(h_max*0.02))
        if h<10:
            h=10

        w_safe = min(df_width - x, x)
        h_safe = min(df_height - y, y)
        w = min(w, w_safe)
        h = min(h, h_safe)

        anchors_list.append([x, y, w, h])

    print(f"结束初始化锚框，生成了 {len(anchors_list)} 个锚框")
    return anchors_list



def generate_initial_anchors(df_origin, num_anchors=50, top_k=50):
    """
    新版初始锚框生成：动态剔除已覆盖区域，从剩余高能量点中采样锚框中心。
    :param df_origin: 输入频谱 DataFrame
    :param num_anchors: 要生成的锚框数量
    :param top_k: 每轮从剩余高能量点中选前 top_k 作为候选
    :return: anchors_list
    """
    print("生成初始锚框...")
    anchors_list = []

    df = df_origin.copy()
    df_height, df_width = df.shape
    background_noise = df.values.min()

    # 初始化边界这一圈尽量不采样，避免刚生成的锚框中心点太贴近频谱边缘
    df.iloc[0:3, :] = background_noise
    df.iloc[-3:-1, :] = background_noise
    df.iloc[:, 0:3] = background_noise
    df.iloc[:, -3:-1] = background_noise

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

        # 动态生成尺寸
        w_max = min(x, df_width - x-1) * 2
        h_max = min(y, df_height - y-1) * 2
        w = max(5, int(random.uniform(0.005, 0.05) * w_max))
        h = max(5, int(random.uniform(0.001, 0.01) * h_max))

        # 截断尺寸防止越界
        w = min(w, df_width - x, x+1)
        h = min(h, df_height - y, y+1)

        anchors_list.append([x, y, w, h])

        # 更新 used_mask，将当前锚框区域标为 True
        x_min = max(0, x - w // 2)
        x_max = min(df_width, x + w // 2)
        y_min = max(0, y - h // 2)
        y_max = min(df_height, y + h // 2)
        used_mask[y_min:y_max, x_min:x_max] = True

    print(f"成功生成 {len(anchors_list)} 个初始锚框。")
    return anchors_list



def generate_more_initial_anchors(df_origin, exist_anchors, num_anchors=50, top_k=50):
    """
    从高能量区域中选择点生成新的初始锚框，避免与已有锚框重叠
    :param df_origin: 原始频谱图 (2D numpy array)
    :param exist_anchors: 已存在的锚框列表，每个为 [cf, ct, w, h]
    :param num_anchors: 最多生成的新锚框数量
    :param top_k: 候选采样点数（从高能量区域中选 top_k 再筛）
    :return: 新生成的锚框列表
    """

    anchors_list = []

    df = df_origin.copy()
    df_height, df_width = df.shape
    background_noise = df.values.min()

    # 初始化边界这一圈尽量不采样，避免刚生成的锚框中心点太贴近频谱边缘
    df.iloc[0:10, :] = background_noise
    df.iloc[-10:-1, :] = background_noise
    df.iloc[:, 0:10] = background_noise
    df.iloc[:, -10:-1] = background_noise

    used_mask = np.zeros_like(df.values, dtype=bool)


    for anchor in exist_anchors:
        x_min = int(max(0, anchor[0] - anchor[2] // 2))
        x_max = int(min(df_width, anchor[0] + anchor[2] // 2))
        y_min = int(max(0, anchor[1] - anchor[3] // 2))
        y_max = int(min(df_height, anchor[1] + anchor[3] // 2))
        used_mask[y_min:y_max, x_min:x_max] = True


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

        # 动态生成尺寸
        w_max = min(x, df_width - x - 1) * 2
        h_max = min(y, df_height - y - 1) * 2
        w = max(15, int(random.uniform(0.005, 0.05) * w_max))
        h = max(10, int(random.uniform(0.001, 0.01) * h_max))

        # 截断尺寸防止越界
        w = min(w, df_width - x, x + 1)
        h = min(h, df_height - y, y + 1)

        anchors_list.append([x, y, w, h])

        # 更新 used_mask，将当前锚框区域标为 True
        x_min = max(0, x - w // 2)
        x_max = min(df_width, x + w // 2)
        y_min = max(0, y - h // 2)
        y_max = min(df_height, y + h // 2)
        used_mask[y_min:y_max, x_min:x_max] = True

    print(f"成功生成 {len(anchors_list)} 个初始锚框。")
    return anchors_list



if __name__=="__main__":
    df_origin = pd.read_csv(
        "dataset/train/DJI_inspire_2_2G_0.csv")
    df=df_origin.iloc[:,1:]
    plot_greyscale_for_singledf(df)
    anchors_list=generate_initial_anchors(df,num_anchors=50)
    plot_greyscale_for_singledf_with_anthor(df,anchors_list,image_name='gray_image_anchor0.png')



