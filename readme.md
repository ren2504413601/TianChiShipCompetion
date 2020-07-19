2020数字中国创新大赛-数字政府赛道-智能算法赛：智慧海洋建设

- 赛题连接：[tianchi ship](<https://tianchi.aliyun.com/competition/entrance/231768/introduction>) 

- AI Studio 平台的实现代码 ： [notebook](<https://aistudio.baidu.com/aistudio/projectdetail/252328>)

- 通过分析渔船北斗设备位置数据，具体判断出是拖网作业、围网作业还是流刺网作业。轨迹（序列数据）+三分类任务，评估指标选用的是F1值

  - 特征选择方法：经纬度统计特征、交叉特征、基于Word2Vec的特征编码（轨迹序列Embedding向量）

  - 模型：LightGBM单模+5折交叉验证

- 排名：初赛107 / 3275、 F1值0.8811，复赛74 / 3275、F1值0.8589