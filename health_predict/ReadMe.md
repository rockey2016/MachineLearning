# 健康模型预测

## 模型描述

**输入：（samples,timesteps_in,input_dim）**

​            samples:样本数

​            timesteps：输入窗口大小

​            input_dim:输入特征维度

**输出： (samples,timesteps_out）**

​           目前仅支持输出为1维的预测(单变量预测)

​           timesteps_out：输出窗口大小

