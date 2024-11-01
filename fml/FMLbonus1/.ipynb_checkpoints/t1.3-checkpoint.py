import numpy as np


def standardize(X):
    """
    标准化数据，使每个特征的均值为0，方差为1。

    Args:
        X: numpy数组，形状为(N, D) 其中N是样本数，D是特征数。

    Returns:
        标准化后的数据。
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def standardize_datasets(xs_train, ys_train, xs_valid, xs_test):
    """
    标准化训练、验证和测试数据集。

    Args:
        xs_train: 训练集特征数据。
        ys_train: 训练集目标数据。
        xs_valid: 验证集特征数据。
        xs_test: 测试集特征数据。

    Returns:
        标准化后的数据集。
    """
    # 标准化训练数据特征和目标
    xs_train_std = standardize(xs_train)
    ys_train_std = ys_train - ys_train.mean()

    # 使用训练数据的统计数据标准化验证和测试数据
    xs_valid_std = standardize(xs_valid)
    xs_test_std = standardize(xs_test)

    return xs_train_std, ys_train_std, xs_valid_std, xs_test_std

# 假设有以下数据集
# xs_train, ys_train, xs_valid, ys_valid, xs_test, ys_test = ...

# 执行标准化
# xs_train_std, ys_train_std, xs_valid_std, xs_test_std = standardize_datasets(xs_train, ys_train, xs_valid, xs_test)

# 输出标准化后的数据集以供检查
# print("标准化后的训练集特征均值:", xs_train_std.mean(axis=0))
# print("标准化后的训练集特征方差:", xs_train_std.std(axis=0))
# print("标准化后的训练集目标均值:", ys_train_std.mean())
# print("标准化后的验证集特征均值:", xs_valid_std.mean(axis=0))
# print("标准化后的验证集特征方差:", xs_valid_std.std(axis=0))
# print("标准化后的测试集特征均值:", xs_test_std.mean(axis=0))
# print("标准化后的测试集特征方差:", xs_test_std.std(axis=0))