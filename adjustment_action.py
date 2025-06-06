'''定义调整的动作'''
import numpy as np
import pandas as pd

def adjust_anchor_box(df_origin,anchor_box, action_type, action_param):
    """
    根据动作类型和参数调整锚框
    参数:
        anchor_box: [center_freq, center_time, width, height]
        action_type: "move", "resize_width", "resize_height", "aspect_ratio"
        action_param: 具体参数
    返回:
        new_anchor_box: 调整后的锚框参数
    """
    cf, ct, w, h = map(float, anchor_box)
    df_h, df_w = df_origin.shape[0], df_origin.shape[1]

    # 设置调整步长比例（相对于宽高）
    move_ratio = 0.2
    resize_ratio = 0.2
    min_size = 6

    if action_type == "move":
        if action_param == "up":
            ct -= max(h * move_ratio, 1)
        elif action_param == "down":
            ct += max(h * move_ratio, 1)
        elif action_param == "left":
            cf -= max(w * move_ratio, 1)
        elif action_param == "right":
            cf += max(w * move_ratio, 1)

    elif action_type == "resize_width":
        if action_param == "shrink":
            w *= (1 - resize_ratio)
        elif action_param == "expand":
            w *= (1 + resize_ratio)

    elif action_type == "resize_height":
        if action_param == "shrink":
            h *= (1 - resize_ratio)
        elif action_param == "expand":
            h *= (1 + resize_ratio)

    # 限制尺寸不能太小
    w = max(min_size, int(w))
    h = max(min_size, int(h))

    # 限制中心不能越界
    cf = np.clip(cf, (w//2)+7, df_w - 7 - w // 2)
    ct = np.clip(ct, (h//2)+7, df_h - 7 - h // 2)

    # 限制尺寸不能超过边界（用于靠近边缘）
    max_w = min(cf * 2, (df_w - cf) * 2)
    max_h = min(ct * 2, (df_h - ct) * 2)
    w = min(w, max_w)
    h = min(h, max_h)

    # 返回整数格式
    return [int(cf), int(ct), int(w), int(h)]


if __name__=="__main__":

    # 使用示例
    df=pd.read_csv("D:\\20250310迹线预处理3_30M\\111\\0_9000MHz原始信号数据\\20250310122051_20250310125051_0.009_900.0081.csv")
    # 示例：向右移动锚框
    new_box = adjust_anchor_box(
        df,
        anchor_box=[100, 200, 50, 30],
        action_type="move",
        action_param="right"
    )
    print("调整后的锚框:", new_box)  # 输出: [105.0, 200, 50, 30]