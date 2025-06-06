import os
import random

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np



from utils.dataset_preprocessing import df_normalization


def plot_greyscale_for_singledf(df, image_name='gray_image_default.png'):
    '''
    为一个单独的csv频谱数据绘制灰度图
    '''
    #print("绘制灰度图......")
    width = df.shape[1]  # 列数对应图片的宽
    height = df.shape[0]  # 行数对应图片的高
    #print("width:" + str(width))
    #print("height:" + str(height))

    # 确定底噪
    all_values = df.values.flatten()
    background_noise = pd.Series(all_values).median()
    #print(f"底噪{background_noise}")

    signal_max = df.values.max()
    #print(f"极大值{signal_max}")

    powerwidth = signal_max - background_noise
    # 使用 Pandas 的向量化操作来设置底噪
    df_clipped = df.clip(lower=background_noise)

    # 计算灰度值并转换为 8 位整数
    gray_values = ((df_clipped - background_noise) / powerwidth * 255).astype(np.uint8)
    # 将 DataFrame 转换为 NumPy 数组
    gray_array = gray_values.values
    # 转换为图像
    img = Image.fromarray(gray_array, 'L')  # 'L' 表示灰度图
    # 保存图片
    img.save(image_name)
    print(f'图片已保存为: {image_name}')
    return img


def background_noise_normalization(df):
    '''噪底归一化'''
    # 确定底噪
    all_values = df.values.flatten()
    background_noise = pd.Series(all_values).median()
    df = df.clip(lower=background_noise)
    return df


def plot_greyscale_for_singledf_with_anthor(df, anchors_list, image_name='gray_image_anchor.png', saveimg=True):
    '''
    为一个单独的csv频谱数据绘制灰度图
    同时也会把锚框绘制上去
    '''
    #print("绘制锚框灰度图......")
    width = df.shape[1]  # 列数对应图片的宽
    height = df.shape[0]  # 行数对应图片的高
    #print("width:" + str(width))
    #print("height:" + str(height))

    # 确定底噪
    all_values = df.values.flatten()
    background_noise = pd.Series(all_values).median()

    signal_max = df.values.max()

    powerwidth = signal_max - background_noise

    # 使用 Pandas 的向量化操作来设置底噪
    df_clipped = df.clip(lower=background_noise)

    # 计算灰度值并转换为 8 位整数
    gray_values = ((df_clipped - background_noise) / powerwidth * 255).astype(np.uint8)
    # 将 DataFrame 转换为 NumPy 数组
    gray_array = gray_values.values
    # 转换为图像
    # 将灰度数组转换为 RGB 模式的图像
    gray_array_rgb = np.stack([gray_array] * 3, axis=-1)  # 复制三个通道
    img = Image.fromarray(gray_array_rgb, 'RGB')
    pixels = img.load()  # 创建像素映射

    for anthor in anchors_list:
        anthor = list(map(int, anthor))
        if anthor[2] > anthor[0] or anthor[2] > df.shape[1] - anthor[0]:
            print("anchor width illegal")
        if anthor[3] > anthor[1] or anthor[3] > df.shape[0] - anthor[1]:
            print("anchor height illegal")

        for i in range(anthor[2]):
            pixels[anthor[0] - int(anthor[2] / 2) + i, anthor[1] - int(anthor[3] / 2)] = (255, 0, 0)
            pixels[anthor[0] - int(anthor[2] / 2) + i, anthor[1] + int(anthor[3] / 2)] = (255, 0, 0)
        for i in range(anthor[3]):
            pixels[anthor[0] - int(anthor[2] / 2), anthor[1] - int(anthor[3] / 2) + i] = (255, 0, 0)
            pixels[anthor[0] + int(anthor[2] / 2), anthor[1] - int(anthor[3] / 2) + i] = (255, 0, 0)
    if saveimg:
        # 如果路径包含目录部分，则创建目录
        if os.path.dirname(image_name):  # 检查路径是否包含目录
            os.makedirs(os.path.dirname(image_name), exist_ok=True)
        # 保存图片
        img.save(image_name)
        print('图片已保存为:' + image_name)
    return img  # 返回img,画gif图用的



def plot_greyscale_for_singledf_with_anthor_and_score(df, anchors_list, image_name='gray_image_anchor.png', saveimg=True):
    '''
    为一个单独的csv频谱数据绘制灰度图
    同时也会把锚框绘制上去
    同时还会把每个锚框对应的评分也绘制上去
    '''
    #print("绘制锚框灰度图......")
    width = df.shape[1]  # 列数对应图片的宽
    height = df.shape[0]  # 行数对应图片的高
    #print("width:" + str(width))
    #print("height:" + str(height))

    # 确定底噪
    all_values = df.values.flatten()
    background_noise = pd.Series(all_values).median()

    signal_max = df.values.max()

    powerwidth = signal_max - background_noise

    # 使用 Pandas 的向量化操作来设置底噪
    df_clipped = df.clip(lower=background_noise)

    # 计算灰度值并转换为 8 位整数
    gray_values = ((df_clipped - background_noise) / powerwidth * 255).astype(np.uint8)
    # 将 DataFrame 转换为 NumPy 数组
    gray_array = gray_values.values
    # 转换为图像
    # 将灰度数组转换为 RGB 模式的图像
    gray_array_rgb = np.stack([gray_array] * 3, axis=-1)  # 复制三个通道
    img = Image.fromarray(gray_array_rgb, 'RGB')
    pixels = img.load()  # 创建像素映射

    for anthor in anchors_list:
        anthor = list(map(int, anthor))
        if anthor[2] > anthor[0] or anthor[2] > df.shape[1] - anthor[0]:
            print("anchor width illegal")
        if anthor[3] > anthor[1] or anthor[3] > df.shape[0] - anthor[1]:
            print("anchor height illegal")

        draw = ImageDraw.Draw(img)
        x0 = anthor[0] - int(anthor[2] / 2)
        y0 = anthor[1] - int(anthor[3] / 2)
        x1 = anthor[0] + int(anthor[2] / 2)
        y1 = anthor[1] + int(anthor[3] / 2)

        # 调节 width 参数来控制边框粗细
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)

        #把对应的锚框评分也标注上去
        draw = ImageDraw.Draw(img)
        text=str(round(__calculate_score(anthor,df),6))
        # 文字位置
        x = anthor[0] - int(anthor[2] / 2)
        y = anthor[1] - int(anthor[3] / 2)-20
        # 设置字体和大小
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # 使用 Arial 字体
        except IOError:
            font = ImageFont.load_default()  # 如果字体文件不存在，使用默认字体
        # 添加文字
        draw.text((x, y), text, fill=(255, 0, 0), font=font)  # 红色文字



    if saveimg:
        # 如果路径包含目录部分，则创建目录
        if os.path.dirname(image_name):  # 检查路径是否包含目录
            os.makedirs(os.path.dirname(image_name), exist_ok=True)
        # 保存图片
        img.save(image_name)
        print('图片已保存为:' + image_name)
    return img  # 返回img,画gif图用的




def plot_greyscale_for_singledf_with_anthor_and_mark(df, anchors_list, mark_list,image_name='gray_image_anchor.png', saveimg=True):
    '''
    为一个单独的csv频谱数据绘制灰度图
    同时也会把锚框绘制上去
    在绘制锚框时，也会在锚框左上角绘制一个标记（这个标记可以是自定义的，需要通过mark_list传进来，这一点与上面的锚框评分不同，那个是自动生成的）
    '''


    # 确定底噪
    all_values = df.values.flatten()
    background_noise = pd.Series(all_values).median()

    signal_max = df.values.max()

    powerwidth = signal_max - background_noise

    # 使用 Pandas 的向量化操作来设置底噪
    df_clipped = df.clip(lower=background_noise)

    # 计算灰度值并转换为 8 位整数
    gray_values = ((df_clipped - background_noise) / powerwidth * 255).astype(np.uint8)
    # 将 DataFrame 转换为 NumPy 数组
    gray_array = gray_values.values
    # 转换为图像
    # 将灰度数组转换为 RGB 模式的图像
    gray_array_rgb = np.stack([gray_array] * 3, axis=-1)  # 复制三个通道
    img = Image.fromarray(gray_array_rgb, 'RGB')
    pixels = img.load()  # 创建像素映射

    for anthor,mark in zip(anchors_list,mark_list):
        anthor = list(map(int, anthor))
        if anthor[2] > anthor[0] or anthor[2] > df.shape[1] - anthor[0]:
            print("anchor width illegal")
        if anthor[3] > anthor[1] or anthor[3] > df.shape[0] - anthor[1]:
            print("anchor height illegal")

        for i in range(anthor[2]):
            pixels[anthor[0] - int(anthor[2] / 2) + i, anthor[1] - int(anthor[3] / 2)] = (255, 0, 0)
            pixels[anthor[0] - int(anthor[2] / 2) + i, anthor[1] + int(anthor[3] / 2)] = (255, 0, 0)
        for i in range(anthor[3]):
            pixels[anthor[0] - int(anthor[2] / 2), anthor[1] - int(anthor[3] / 2) + i] = (255, 0, 0)
            pixels[anthor[0] + int(anthor[2] / 2), anthor[1] - int(anthor[3] / 2) + i] = (255, 0, 0)
        #把对应的锚框标记也标注上去
        draw = ImageDraw.Draw(img)
        text=str(mark)
        # 文字位置
        x = anthor[0] - int(anthor[2] / 2)
        y = anthor[1] - int(anthor[3] / 2)-20
        # 设置字体和大小
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # 使用 Arial 字体
        except IOError:
            font = ImageFont.load_default()  # 如果字体文件不存在，使用默认字体
        # 添加文字
        draw.text((x, y), text, fill=(255, 0, 0), font=font)  # 红色文字



    if saveimg:
        # 如果路径包含目录部分，则创建目录
        if os.path.dirname(image_name):  # 检查路径是否包含目录
            os.makedirs(os.path.dirname(image_name), exist_ok=True)
        # 保存图片
        img.save(image_name)
        print('图片已保存为:' + image_name)
    return img  # 返回img,画gif图用的







def save_as_gif(img_list, duration=500, gif_name="default.gif"):
    '''
    将图像列表保存为 GIF 动画
    :param img_list: 图像对象列表
    :param duration: 每帧的持续时间（毫秒）
    :param gif_name: 保存的 GIF 文件名
    '''
    print("保存 GIF 动画......")
    # 保存为 GIF 动画
    add_start_text = True
    if add_start_text and len(img_list) > 0:
        # 在第一个图像中添加“start”文字
        img = img_list[0].copy()  # 复制第一个图像
        draw = ImageDraw.Draw(img)

        # 设置字体和大小
        try:
            font = ImageFont.truetype("arial.ttf", 20)  # 使用 Arial 字体，大小为 20
        except IOError:
            font = ImageFont.load_default()  # 如果字体文件不存在，使用默认字体

        # 获取文字尺寸
        text = "start"

        # 计算文字位置（中央）
        '''img_width, img_height = img.size
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2'''

        # 文字位置（左上）
        x = 0
        y = 0

        # 添加文字
        draw.text((x, y), text, fill=(255, 0, 0), font=font)  # 红色文字
        img_list[0] = img  # 替换第一个图像

        # 接下来给每一个图像添加一个编号
        for i in range(1, len(img_list)):
            img = img_list[i].copy()
            draw = ImageDraw.Draw(img)
            # 设置字体和大小
            try:
                font = ImageFont.truetype("arial.ttf", 30)  # 使用 Arial 字体，大小为 20
            except IOError:
                font = ImageFont.load_default()  # 如果字体文件不存在，使用默认字体
            text = f"{i}/{len(img_list)-1}"
            x = 0
            y = 0

            # 添加文字
            draw.text((x, y), text, fill=(0, 255, 0), font=font)  # 红色文字
            img_list[i] = img


    img_list[0].save(
        gif_name,
        save_all=True,
        append_images=img_list[1:],
        duration=duration,
        loop=0  # 无限循环
    )
    print(f"GIF 动画已保存为: {gif_name}")


def __calculate_score(anchor_box, df_origin):
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
    region_inner = df_origin.iloc[t_min:t_max+1, f_min:f_max+1].values

    # 计算统计量
    mean_inner = np.mean(region_inner) if region_inner.size > 0 else 0.0


    # 评分项计算
    score1 = mean_inner
    # 最终评分
    final_score =score1
    return float(final_score)


def __generate_initial_anchors(df_origin, num_anchors=50, top_k=50):
    """
    （这个函数是这里测试用的）
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

        # 动态生成尺寸
        w_max = min(x, df_width - x) * 2
        h_max = min(y, df_height - y) * 2
        w = max(15, int(random.uniform(0.05, 0.2) * w_max))
        h = max(10, int(w * 0.2))

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

    print(f"成功生成 {len(anchors_list)} 个初始锚框。")
    return anchors_list

if __name__ == "__main__":
    print("start test...")
    df = pd.read_csv("../dataset/train/DJI_inspire_2_2G_0.csv")
    df = df.iloc[:, 1:]
    df=df_normalization(df)
    anchors_list=__generate_initial_anchors(df,num_anchors=20)
    plot_greyscale_for_singledf_with_anthor_and_score(df,anchors_list,image_name="g12.png")
