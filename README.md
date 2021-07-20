# Laplace_Operator_on_Mesh
计算mesh上的拉普拉斯贝尔特拉米算子（python版，运行速度同matlab相同甚至稍快）

通过广义特征值问题进行求解
$$
W\phi = \lambda A\phi
$$


## 结果
![result](https://github.com/wanghuzhen/Laplace_Operator_on_Mesh/blob/main/result.PNG)

### 更大型矩阵将numpy替换为cupy，支持CUDA加速