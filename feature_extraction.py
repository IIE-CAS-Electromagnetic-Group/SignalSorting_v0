import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from calculate_NLL import fit_noise_gmm
from calculate_reward import calculate_score_by_nll, calculate_score
from global_features_extract import generate_noise_anchors, generate_signal_anchors, get_global_noise_nll, \
    get_global_signal_nll
from init_anchor import generate_initial_anchors
from utils.dataset_preprocessing import df_normalization
from utils.plot_greyscale import background_noise_normalization, plot_greyscale_for_singledf_with_anthor
from scipy.ndimage import zoom




class EnhancedStateEncoderWithClassifier(nn.Module):
    def __init__(self, output_dim=64, num_classes=2):
        super().__init__()
        # 特征提取部分
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 增加通道数
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # 特征映射到固定维度
        self.fc_feature = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        features = self.fc_feature(x)
        logits = self.classifier(features)
        return features, logits

    def extract_features(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        features = self.fc_feature(x)
        return features



# 定义数据集类
class SpectrogramDataset(Dataset):
    def __init__(self, df_noise_list, df_signal_list):
        self.noise_data = df_noise_list
        self.signal_data = df_signal_list
        # 标签：噪声为0，信号为1
        self.labels = [0] * len(self.noise_data) + [1] * len(self.signal_data)
        # 合并数据
        self.data = self.noise_data + self.signal_data

        # 打乱数据和标签,如果训练数据不打乱，模型可能在训练初期就拟合了数据的顺序。
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据和标签
        spectrogram = self.data[idx]
        label = self.labels[idx]
        # 转换为张量
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        label_tensor = torch.tensor(label, dtype=torch.long)
        return spectrogram_tensor, label_tensor

def resize_spectrogram(spectrogram, target_shape=(64, 64)):
    """
    调整频谱图大小到目标尺寸
    参数:
        spectrogram: 输入的频谱图（NumPy 数组）
        target_shape: 目标尺寸（高度，宽度）
    返回:
        调整尺寸后的频谱图
    """
    zoom_factor = (target_shape[0] / spectrogram.shape[0], target_shape[1] / spectrogram.shape[1])
    resized_spectrogram = zoom(spectrogram, zoom_factor, mode='reflect')
    return resized_spectrogram

# 训练函数
def train_model(df_origin_list, global_feature_lists=None,num_epochs=10, batch_size=32, validation_split=0.2):
    arr_noise_list=[]
    arr_signal_list=[]

    #先生成噪声和信号锚框
    num=0
    for index,df in enumerate(df_origin_list):
        noise_anchors_list=generate_noise_anchors(df, num_anchors=30,anchor_size=64)
        signal_anchors_list=generate_signal_anchors(df,num_anchors=30,anchor_size=64)

        '''if global_feature_lists==None:
            # 高斯混合模型建模
            gmm_model = fit_noise_gmm(df, n_components=3, remove_top_percent=1.0)
            noise_nll = get_global_noise_nll(df)  # 噪声参考值
            signal_nll = get_global_signal_nll(df)  # 信号参考值
        else:
            gmm_model=global_feature_lists[index][0]
            noise_nll=global_feature_lists[index][1]
            signal_nll=global_feature_lists[index][2]

        for noise_anchor in noise_anchors_list:
            if calculate_score_by_nll(noise_anchor,df,gmm_model=gmm_model,noise_nll=noise_nll,signal_nll=signal_nll)>0.01:
                noise_anchors_list.remove(noise_anchor)#删除掉评分过高的噪声锚框
        for signal_anchor in signal_anchors_list:
            if calculate_score_by_nll(signal_anchor,df,gmm_model=gmm_model,noise_nll=noise_nll,signal_nll=signal_nll)<0.9:
                signal_anchors_list.remove(signal_anchor)#删除掉评分过低的信号锚框'''

        for noise_anchor in noise_anchors_list:
            if calculate_score(noise_anchor, df) > 0.05:
                noise_anchors_list.remove(noise_anchor)  # 删除掉评分过高的噪声锚框
        for signal_anchor in signal_anchors_list:
            if calculate_score(signal_anchor, df) < 0.4:
                signal_anchors_list.remove(signal_anchor)  # 删除掉评分过低的信号锚框

        plot_greyscale_for_singledf_with_anthor(df,noise_anchors_list,f"images/cnn_train/xunlian{num}_噪声.png")
        plot_greyscale_for_singledf_with_anthor(df, signal_anchors_list, f"images/cnn_train/xunlian{num}_信号.png")
        num=num+1

        for noise_anchor in noise_anchors_list:

            # 提取噪声锚框区域
            cf, ct, w, h = noise_anchor
            f_min = int(cf - w / 2)
            f_max = int(cf + w / 2)
            t_min = int(ct - h / 2)
            t_max = int(ct + h / 2)

            # 边界检查
            f_min = max(0, f_min)
            f_max = min(df.shape[1], f_max)
            t_min = max(0, t_min)
            t_max = min(df.shape[0], t_max)

            df_noise = df.iloc[t_min:t_max + 1, f_min:f_max + 1]
            arr_noise_list.append(resize_spectrogram(df_noise.values))

        for signal_anchor in signal_anchors_list:
            # 提取噪声锚框区域
            cf, ct, w, h = signal_anchor
            f_min = int(cf - w / 2)
            f_max = int(cf + w / 2)
            t_min = int(ct - h / 2)
            t_max = int(ct + h / 2)

            # 边界检查
            f_min = max(0, f_min)
            f_max = min(df.shape[1], f_max)
            t_min = max(0, t_min)
            t_max = min(df.shape[0], t_max)

            df_signal = df.iloc[t_min:t_max + 1, f_min:f_max + 1]
            arr_signal_list.append(resize_spectrogram(df_signal.values))


    # 创建数据集
    dataset = SpectrogramDataset(arr_noise_list, arr_signal_list)
    # 划分训练集和验证集
    train_data, val_data, train_labels, val_labels = train_test_split(
        dataset.data, dataset.labels, test_size=validation_split, random_state=42)
    train_dataset = SpectrogramDataset(train_data, [])
    val_dataset = SpectrogramDataset(val_data, [])


    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedStateEncoderWithClassifier(output_dim=64, num_classes=2).to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # 计算训练集的平均损失和准确率
        train_loss /= len(train_loader)
        train_acc = 100. * correct / total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                _, outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # 计算验证集的平均损失和准确率
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 打印训练和验证信息
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # 返回训练好的模型
    return model


# 测试函数
def test_model(model, df_origin_list, batch_size=32):
    arr_noise_list = []
    arr_signal_list = []

    num=0
    for df in df_origin_list:
        noise_anchors_list = generate_noise_anchors(df, num_anchors=10, anchor_size=64)
        signal_anchors_list = generate_signal_anchors(df, num_anchors=10, anchor_size=64)

        for noise_anchor in noise_anchors_list:
            if calculate_score(noise_anchor, df) > 0.05:
                noise_anchors_list.remove(noise_anchor)  # 删除掉评分过高的噪声锚框
        for signal_anchor in signal_anchors_list:
            if calculate_score(signal_anchor, df) < 0.4:
                signal_anchors_list.remove(signal_anchor)  # 删除掉评分过低的信号锚框


        plot_greyscale_for_singledf_with_anthor(df, noise_anchors_list, f"images/cnn_train/ceshi{num}_噪声.png")
        plot_greyscale_for_singledf_with_anthor(df, signal_anchors_list, f"images/cnn_train/ceshi{num}_信号.png")
        num =num+1

        for noise_anchor in noise_anchors_list:
            # 提取噪声锚框区域
            cf, ct, w, h = noise_anchor
            f_min = int(cf - w / 2)
            f_max = int(cf + w / 2)
            t_min = int(ct - h / 2)
            t_max = int(ct + h / 2)

            # 边界检查
            f_min = max(0, f_min)
            f_max = min(df.shape[1], f_max)
            t_min = max(0, t_min)
            t_max = min(df.shape[0], t_max)

            df_noise = df.iloc[t_min:t_max + 1, f_min:f_max + 1]
            arr_noise_list.append(resize_spectrogram(df_noise.values))

        for signal_anchor in signal_anchors_list:
            # 提取噪声锚框区域
            cf, ct, w, h = signal_anchor
            f_min = int(cf - w / 2)
            f_max = int(cf + w / 2)
            t_min = int(ct - h / 2)
            t_max = int(ct + h / 2)

            # 边界检查
            f_min = max(0, f_min)
            f_max = min(df.shape[1], f_max)
            t_min = max(0, t_min)
            t_max = min(df.shape[0], t_max)

            df_signal = df.iloc[t_min:t_max + 1, f_min:f_max + 1]
            arr_signal_list.append(resize_spectrogram(df_signal.values))


    # 创建测试数据集
    test_dataset = SpectrogramDataset(arr_noise_list, arr_signal_list)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            _, outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")




def get_safe_region(region,min_size=4):
    """
    提取安全的锚框区域，确保尺寸不低于min_size
    参数:
        min_size: 最小允许尺寸（默认4x4）
    返回:
        region: 填充后的区域（至少为min_size x min_size）
    """
    # 如果原始区域尺寸过小，进行填充
    pad_h = max(min_size - region.shape[0], 0)
    pad_w = max(min_size - region.shape[1], 0)

    background_noise=-9

    # 对称填充
    region = np.pad(region,
                    ((pad_h // 2, pad_h - pad_h // 2),
                     (pad_w // 2, pad_w - pad_w // 2)),
                    mode='constant',
                    constant_values=background_noise)
    #print(region)
    return region




def get_feature_old(anchor_box,df_origin,feature_extraction_model):

    # 确保anchor_box是一个列表或数组
    if isinstance(anchor_box, torch.Tensor):
        anchor_box = anchor_box.tolist()

    # 将df_origin张量转换为DataFrame
    if isinstance(df_origin, torch.Tensor):
        df_origin_cpu=df_origin.cpu()
        # 确保张量是二维的
        if df_origin_cpu.dim() == 3 and df_origin_cpu.shape[0] == 1:
            df_origin_cpu = df_origin_cpu.squeeze(0)  # 移除第一个维度
        df=pd.DataFrame(df_origin_cpu)
    else:
        df=df_origin
    # 提取锚框内部区域
    '''f_min = max(0, int(anchor_box[0] - anchor_box[2] / 2))
    f_max = min(df_origin.shape[0], int(anchor_box[0] + anchor_box[2] / 2))
    t_min = max(0, int(anchor_box[1] - anchor_box[3] / 2))
    t_max = min(df_origin.shape[1], int(anchor_box[1] + anchor_box[3] / 2))'''

    # 提取锚框内部和周围区域
    f_min = max(0, int(anchor_box[0] - 0.7*anchor_box[2]))
    f_max = min(df_origin.shape[1], int(anchor_box[0] + 0.7*anchor_box[2]))
    t_min = max(0, int(anchor_box[1] - 0.7*anchor_box[3]))
    t_max = min(df_origin.shape[0], int(anchor_box[1] + 0.7*anchor_box[3]))

    df = df.iloc[f_min:f_max+1, t_min:t_max+1]
    df = get_safe_region(df, min_size=4)
    # 2. 转换为PyTorch张量并添加通道维度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region_tensor = torch.tensor(df, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    #model = FlexibleStateEncoder(output_dim=output_dim).to(device)
    # 3. 通过模型提取特征
    with torch.no_grad():
        state = feature_extraction_model.extract_features(region_tensor)

    region_feature=state.cpu().numpy().flatten()#锚框内以及周围区域的细粒度特征向量
    overall_feature=np.array(anchor_box)#锚框的整体特征，包括中心点、宽、高。
    return np.concatenate((region_feature,overall_feature),axis=0)


def compress_spectrum(df, target_shape=(8, 8), fill_value=0):
    """
    将任意尺寸的能量矩阵 DataFrame 压缩为 8x8（或指定大小）。

    参数:
        df (pd.DataFrame): 任意尺寸的频谱区域数据
        target_shape (tuple): 默认目标为 (8, 8)
        fill_value (float): 若 df 过小需填充时使用的默认值

    返回:
        pd.DataFrame: 压缩后的 8x8 DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是 pandas.DataFrame")
    if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
        print("警告：提取区域尺寸太小")
    original = df.values.astype(np.float32)
    h, w = original.shape
    target_h, target_w = target_shape

    # 对称填充（适用于小尺寸情况）
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        original = np.pad(original,
                          ((pad_h // 2, pad_h - pad_h // 2),
                           (pad_w // 2, pad_w - pad_w // 2)),
                          mode='constant',
                          constant_values=fill_value)

    # 插值缩放至目标尺寸
    zoom_factors = (target_h / original.shape[0], target_w / original.shape[1])
    compressed = zoom(original, zoom=zoom_factors, order=1)  # 线性插值

    return pd.DataFrame(compressed)

def get_feature(anchor_box, df_origin, feature_extraction_model=None):
    """
    提取锚框区域特征 + anchor自身参数，构成状态向量（长度=64+4=68）
    已加入完整防御逻辑：NaN检查、边界检查、空区域拦截等。
    """
    # 确保 anchor_box 是列表
    if isinstance(anchor_box, torch.Tensor):
        anchor_box = anchor_box.tolist()

    # 将 df_origin 转为 DataFrame
    if isinstance(df_origin, torch.Tensor):
        df_origin_cpu = df_origin.cpu()
        if df_origin_cpu.dim() == 3 and df_origin_cpu.shape[0] == 1:
            df_origin_cpu = df_origin_cpu.squeeze(0)
        df = pd.DataFrame(df_origin_cpu)
    else:
        df = df_origin

    # 提取锚框附近区域（0.7 倍宽高）
    f_min = max(0, int(anchor_box[0] - 0.7 * anchor_box[2]))
    f_max = min(df.shape[1] - 1, int(anchor_box[0] + 0.7 * anchor_box[2]))
    t_min = max(0, int(anchor_box[1] - 0.7 * anchor_box[3]))
    t_max = min(df.shape[0] - 1, int(anchor_box[1] + 0.7 * anchor_box[3]))

    # 提取子区域
    df_region = df.iloc[t_min:t_max + 1, f_min:f_max + 1]

    # 空区域判定（无效 anchor）
    if df_region.empty or df_region.isnull().values.all():
        print("⚠️ 锚框区域为空或无效！", anchor_box)
        return np.zeros(68, dtype=np.float32)

    # 压缩为 8×8 区域特征
    df_compressed = compress_spectrum(df_region, fill_value=0)

    # 转为一维区域特征向量
    region_feature = df_compressed.values.flatten().astype(np.float32)

    #对于锚框本身的宏观特征，比如位置信息，长宽等等，先归一化一下再拼接到状态向量里
    center_freq_norm=float(anchor_box[0])/df_origin.shape[1]
    center_time_norm=float(anchor_box[1])/df_origin.shape[0]
    freq_width_norm=float(anchor_box[2])/df_origin.shape[1]
    time_height_norm=float(anchor_box[3])/df_origin.shape[0]
    # 拼接 anchor 参数
    anchor_feature = np.array([center_freq_norm,center_time_norm,freq_width_norm,time_height_norm], dtype=np.float32)

    # 拼接状态向量
    state_vector = np.concatenate([region_feature, anchor_feature], axis=0)

    # NaN/Inf 清洗
    if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
        print("⚠️ 状态向量包含非法值！已进行清洗。")
        print("原 anchor：", anchor_box)
        print("区域原始值：\n", df_region.values)
        print("压缩后：\n", df_compressed.values)
        state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=10.0, neginf=-10.0)

    return state_vector

# 保存 t-SNE 用的特征和标签
def extract_features_for_tsne(model, df_list, save_prefix="tsne"):
    model.eval()
    features = []
    labels = []
    for df in df_list:
        noise_anchors = generate_noise_anchors(df, 30,anchor_size=64)
        signal_anchors = generate_signal_anchors(df, 30,anchor_size=64)

        for label, anchor_list in zip([0, 1], [noise_anchors, signal_anchors]):
            for anchor in anchor_list:
                feat = get_feature(anchor, df, model)
                features.append(feat)
                labels.append(label)

    features = torch.tensor(features)
    labels = torch.tensor(labels)

    torch.save(features, f"{save_prefix}_features.pt")
    torch.save(labels, f"{save_prefix}_labels.pt")
    print("Features saved for t-SNE visualization.")

if __name__=="__main__":

    '''print("----------------加载训练集--------------------")
    dataset_path = "dataset/train"  # 数据集文件夹路径
    df_origin_list = []

    # 遍历数据集文件夹，加载每个csv文件并生成每个csv频谱文件的初始锚框，并且生成每个csv文件的全局特征
    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(dataset_path, file_name)
            print(f"正在加载文件: {file_path}")
            df_origin = pd.read_csv(file_path)

            # 取部分区域数据(去掉第一列时间列)，并进行噪声归一化
            df = df_origin.iloc[:, 1:]
            df=df_normalization(df)

            # plot_greyscale_for_singledf(df, file_name+"_可视化.png")
            df_origin_list.append(df)


    trained_model = train_model(df_origin_list, num_epochs=200, batch_size=32)

    torch.save(trained_model, "weights/cnn.pth")

    test_model(trained_model, df_origin_list)

    
    # 使用示例
    df_origin = pd.read_csv(
        "dataset/train/DJI_inspire_2_2G_0.csv")
    anchor_list=generate_initial_anchors(df_origin,num_anchors=5)
    plot_greyscale_for_singledf_with_anthor(df_origin,anchor_list)





    print(get_feature(anchor_list[0],df_origin,feature_extraction_model=trained_model))
    print(get_feature(anchor_list[1], df_origin,feature_extraction_model=trained_model))

    model = torch.load("weights/cnn.pth", weights_only=False)

    extract_features_for_tsne(model, df_origin_list, save_prefix="tsne")'''
    df_origin = pd.read_csv(
        "dataset/train/DJI_inspire_2_2G_0.csv")
    anchor_list = generate_initial_anchors(df_origin, num_anchors=5)
    print(get_feature(anchor_list[1], df_origin))


