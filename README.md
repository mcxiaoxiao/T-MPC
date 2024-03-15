# 电科院预测模型代码说明

- 修正模型
    - fydyc_dataset.py 修正模型 数据集类
    - fydyc 数据 csv文档
    - fydyc.ipynb 训练+保存修正模型
    - demo_fydyc.ipynb 预测修正模型效果演示

- 厂用率预测
    - power_plant_dataset.py 厂用率预测 数据集类
    - fdcl/split 拆分开的各发电类型数据集 csv文档
    - fdcl.ipynb 厂用率预测 训练+保存模型（其中dataset = PowerPlantDataset(csv_file='fdcl/split/df_gf.csv')填写训练数据集地址）
    - demo_fdcl.ipynb 厂用率预测模型效果演示