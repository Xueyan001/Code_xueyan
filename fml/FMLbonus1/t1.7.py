from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 假设的标准化训练和验证数据，这些应当被替换为实际的数据
# 示例训练数据集特征 (输入)
xs_train = np.array([[2, 3], [1, 2], [3, 5], [5, 8]])
# 示例训练数据集目标 (输出)
ys_train = np.array([10, 8, 14, 26])

# 示例验证数据集特征 (输入)
xs_valid = np.array([[1, 4], [2, 1], [3, 3], [4, 5]])
# 示例验证数据集目标 (输出)
ys_valid = np.array([12, 6, 10, 18])

# 初始化多项式特征生成器，最高三次方
poly = PolynomialFeatures(degree=3)

# 训练数据：生成多项式特征
xs_train_poly = poly.fit_transform(xs_train)
# 验证数据：生成多项式特征（使用与训练数据相同的转换）
xs_valid_poly = poly.transform(xs_valid)

# 创建线性回归模型
model = LinearRegression()

# 使用多项式特征训练模型
model.fit(xs_train_poly, ys_train)

# 使用训练好的模型在验证数据上进行预测
ys_pred = model.predict(xs_valid_poly)

# 计算在验证数据上的均方误差（MSE）
mse = mean_squared_error(ys_valid, ys_pred)
mse, ys_pred  # 返回MSE值和验证数据集上的预测值