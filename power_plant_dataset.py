import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 定义自定义数据集类
class PowerPlantDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # 加载CSV文件
        self.data = pd.read_csv(csv_file)
        # 初始化变换（如果有的话）
        self.transform = transform

    def __len__(self):
        # 返回数据集中的样本数量
        return len(self.data)

    def __getitem__(self, idx):
        # 从数据集中获取一项数据
        sample = self.data.iloc[idx]
        # 提取特征和标签
        features = torch.tensor([
            sample['发电出力值'],  # 发电出力值
            pd.to_datetime(sample['日期时间']).timestamp()  # 转换日期时间为时间戳
        ], dtype=torch.float32)
        label = torch.tensor(sample['厂用率'], dtype=torch.float32)

        # 应用变换（如果有的话）
        if self.transform:
            features = self.transform(features)

        return features, label