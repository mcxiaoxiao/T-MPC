import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from fydyc_dataset import FydycDataset
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
class PolynomialRegression1(nn.Module):
    def __init__(self, degree):
        super(PolynomialRegression1, self).__init__()
        self.degree = degree
        # 计算两个变量的多项式特征数量（包括交叉项和常数项）
        num_features = (degree + 1) * (degree + 2) // 2
        self.linear = nn.Linear(num_features, 1)
    
    def forward(self, x):
        # x应该是一个二维张量，其中x[:, 0]是x1，x[:, 1]是x2
        poly_features = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                # 通过使用unsqueeze方法增加一个维度，确保每个特征都是二维的
                poly_feature = (x[:, 0] ** i * x[:, 1] ** j).unsqueeze(1)
                poly_features.append(poly_feature)
        poly_features = torch.cat(poly_features, dim=1)
        return self.linear(poly_features)

        
degree = 3  # 3次多项式
model = PolynomialRegression1(degree)
model.load_state_dict(torch.load('fydyc/1000epoch-供电-多项式回归-3次.pth'))
dataset = FydycDataset(csv_file='fydyc/供电.csv')
csv_file = 'fydyc/供电_valid.csv'
data = pd.read_csv(csv_file)
import torch
import numpy as np
from datetime import datetime

# 加载模型
modelfl = PolynomialRegression1(degree)
modelfl.load_state_dict(torch.load('fydyc/400epoch-风力-多项式回归-3次.pth'))
modelgf = PolynomialRegression1(degree)
modelgf.load_state_dict(torch.load('fydyc/1000epoch-光伏-多项式回归-3次.pth'))
modelgd = PolynomialRegression1(degree)
modelgd.load_state_dict(torch.load('fydyc/1000epoch-供电-多项式回归-3次.pth'))

models = {'风力': modelfl, '光伏': modelgf, '供电': modelgd}
datas = {'风力': 'fydyc/风力.csv', '光伏': 'fydyc/光伏.csv', '供电': 'fydyc/供电.csv'}

def predict(value, current_date, predict_date, model_name):
    """
    根据输入的预测值、当前日期、预测日期和模型名称进行预测。
    
    参数:
    - value: 预测值（float）
    - current_date: 当前日期（str，格式为'YYYY-MM-DD'）
    - predict_date: 预测日期（str，格式为'YYYY-MM-DD'）
    - model_name: 模型名称（str，可选项为'风力'、'光伏'、'供电'）
    
    返回:
    - 预测结果（float）
    """
    dataset = FydycDataset(csv_file = datas[model_name])
    
    if value == 0:
        return 0.00
    
    # 转换日期格式
    current_date = datetime.strptime(current_date, '%Y%m%d').date()
    predict_date = datetime.strptime(predict_date, '%Y%m%d').date()
    
    # 计算日期差
    date_diff = (predict_date - current_date).days
    
    # 归一化输入
    normalized_input = dataset.scaler_features.transform([[value, date_diff]])
    input_tensor = torch.from_numpy(normalized_input).float()
    
    # 选择模型并进行预测
    selected_model = models[model_name]
    with torch.no_grad():
        output_tensor = selected_model(input_tensor)
    
    # 逆归一化输出值
    output_value = output_tensor.numpy()
    predicted_value = dataset.scaler_labels.inverse_transform(output_value)
    
    return predicted_value[0][0]


from power_plant_dataset import PowerPlantDataset

# 定义模型
class PolynomialRegressor(nn.Module):
    def __init__(self, degree):
        super(PolynomialRegressor, self).__init__()
        self.degree = degree
        self.coefficients_mean = nn.Parameter(torch.randn(degree + 1))
        self.coefficients_std = nn.Parameter(torch.randn(degree + 1))

    def forward(self, x):
        mean = sum(self.coefficients_mean[i] * x.pow(i) for i in range(self.degree + 1))
        std = torch.exp(sum(self.coefficients_std[i] * x.pow(i) for i in range(self.degree + 1)))
        return mean, std

def predict_power_output(power_output, power_type):
    # 置信区间配置
    confidence_intervals = {
        'fd': {'lower': 2.0, 'upper': 2.0},
        'gf': {'lower': 1.5, 'upper': 2.5},
        'hdrm': {'lower': 1.2, 'upper': 1.2},
        'sdyg': {'lower': 0.5, 'upper': 1.0}
    }
    
    # 数据和模型路径配置
    dataset_file = f'fdcl/split/df_{power_type}.csv'
    model_file = f'fdcl/{power_type}.pth'
    
    # 加载数据集
    dataset = PowerPlantDataset(csv_file=dataset_file)
    
    # 初始化模型
    model = PolynomialRegressor(degree=3)
    
    # 加载模型参数
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
    else:
        print(f"模型参数文件 '{model_file}' 不存在.")
        return
    
    # 数据预处理
    normalized_input = dataset.scaler_features.transform([[power_output]])
    input_tensor = torch.from_numpy(normalized_input).float()
    
    # 进行预测
    with torch.no_grad():
        mean, std = model(input_tensor)
    
    # 逆归一化输出值
    output_mean = mean.numpy()
    output_std = std.numpy()
    predicted_mean = dataset.scaler_labels.inverse_transform(output_mean.reshape(1, -1))
    predicted_std = dataset.scaler_labels.inverse_transform(output_std.reshape(1, -1))
    
    # 计算置信区间
    ci = confidence_intervals[power_type]
    lower_bound = dataset.scaler_labels.inverse_transform((mean - ci['lower'] * std).numpy().reshape(1, -1))
    upper_bound = dataset.scaler_labels.inverse_transform((mean + ci['upper'] * std).numpy().reshape(1, -1))
    
    print(f"Predicted Mean: {predicted_mean}")
    print(f"Predicted Std Dev: {predicted_std}")
    print(f"Confidence Interval: {lower_bound} to {upper_bound}")
    return {"mean":predicted_mean[0][0],"std":predicted_std[0][0],"interval_lower":lower_bound[0][0],"interval_upper":upper_bound[0][0]}

# 导入 FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

# 创建 FastAPI 实例
app = FastAPI()

class PredictionResult(BaseModel):
    mean: float
    std: float
    interval_lower: float
    interval_upper: float


# 创建一个路由
@app.get("/cylpredict", response_model=PredictionResult)
async def predict_cyl(value: float, power_type: str):
    # 调用 predict 函数并返回结果
    predicted_result = predict_power_output(value, power_type)
    print(f'厂用率预测: {predicted_result}')
    return predicted_result


# 创建一个路由
@app.get("/predict")
async def predict_endpoint(value: float, start_date: str, end_date: str, power_type: str):
    # 调用 predict 函数并返回结果
    predicted_result = predict(value, start_date, end_date, power_type)
    print(f'修正后的预测结果: {predicted_result}')
    return predicted_result


