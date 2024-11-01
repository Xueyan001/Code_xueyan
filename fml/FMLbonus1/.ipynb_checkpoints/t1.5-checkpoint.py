import numpy as np


def my_quadratic_features(xs: np.ndarray) -> np.ndarray:
    """
    为给定数据生成二次多项式特征。

    Args:
    xs: 一个形状为(N, D)的2D numpy数组，包含N个D维样本。

    Returns:
    一个(N, M)的numpy数组，包含了转换后的输入。
    """
    # 获取样本数量和特征维度
    N, D = xs.shape

    # 初始化特征列表，包含偏置项
    features = [np.ones((N, 1))]  # 偏置项

    # 添加一次项和二次项
    for d in range(D):
        features.append(xs[:, d:d + 1])  # 一次项
        features.append(xs[:, d:d + 1] ** 2)  # 二次项

        # 交互项，避免重复
        for d2 in range(d + 1, D):
            features.append(xs[:, d:d + 1] * xs[:, d2:d2 + 1])

    # 横向合并所有特征
    return np.hstack(features)

# 示例数据
# 假设有一个形状为(N, 2)的输入数组
# xs = np.array([[1, 2], [3, 4], [5, 6]])

# 使用函数计算二次多项式特征
# quadratic_features = my_quadratic_features(xs)

# 输出结果
# print(quadratic_features)

# 注意：在实际应用中，你需要将 `xs` 替换为你的实际数据集
