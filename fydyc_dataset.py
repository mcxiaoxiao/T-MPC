import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FydycDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # 加载CSV文件
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # 创建标准化器
        self.scaler_features = MinMaxScaler()
        self.scaler_labels = MinMaxScaler()

        # 提取特征和标签
        labels = self.data['实际值'].values.reshape(-1, 1)
        # 现在features包括两个特征：预测值和间隔天数
        features = self.data[['预测值', '间隔天数']].values

        # 训练标准化器并应用于特征和标签
        self.features = self.scaler_features.fit_transform(features)
        self.labels = self.scaler_labels.fit_transform(labels)

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.data)

    def __getitem__(self, idx):
        # 从数据集中获取一项数据
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

    def inverse_transform_features(self, values):
        # 逆向标准化特征
        return self.scaler_features.inverse_transform(values)

    def inverse_transform_labels(self, values):
        # 逆向标准化标签
        return self.scaler_labels.inverse_transform(values.reshape(-1, 1))
