import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 定义自定义数据集类
class FydycDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # 加载CSV文件
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # 创建标准化器
        self.scaler_features = MinMaxScaler()
        self.scaler_labels = MinMaxScaler()

        # 提取特征和标签
        features = self.data['风电预测'].values.reshape(-1, 1)+self.data['光伏预测'].values.reshape(-1, 1)+self.data['火电计划'].values.reshape(-1, 1)+self.data['水电计划'].values.reshape(-1, 1)
        labels = self.data['供电预测'].values.reshape(-1, 1)

        # 训练标准化器并应用于特征和标签
        self.data['总预测'] = self.scaler_features.fit_transform(features)
        self.data['供电预测'] = self.scaler_labels.fit_transform(labels)

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.data)

    def __getitem__(self, idx):
        # 从数据集中获取一项数据
        sample = self.data.iloc[idx]
        # 提取特征和标签
        features = torch.tensor(sample['总预测'], dtype=torch.float32)
        label = torch.tensor(sample['供电预测'], dtype=torch.float32)
        return features, label

    def inverse_transform_features(self, values):
        # 逆向标准化特征
        return self.scaler_features.inverse_transform(values)

    def inverse_transform_labels(self, values):
        # 逆向标准化标签
        return self.scaler_labels.inverse_transform(values)

