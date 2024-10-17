import pandas as pd
from imblearn.under_sampling import ClusterCentroids, NearMiss
from pandas.core.common import random_state
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 假设数据集为 'data_transformed.csv'
data = pd.read_csv('../data_transformed1.csv')

# 特征和目标变量
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# 列出不需要标准化的特征
non_scaled_features = ['Net Income Flag', 'Liability-Assets Flag']

# 初始化 KFold，n_splits=5 表示5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化ClusterCentroids
cluster_centroids = ClusterCentroids(random_state=42)
near_miss=NearMiss(n_neighbors=3)

# 定义模型
model = LogisticRegression(max_iter=1000)

# 保存每次迭代的准确率和平衡准确率
accuracy_scores = []
balanced_accuracy_scores = []

# 开始交叉验证过程
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    # 划分训练集和测试集
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 分别处理需要标准化和不需要标准化的特征
    X_train_scaled = X_train.drop(non_scaled_features, axis=1)
    X_test_scaled = X_test.drop(non_scaled_features, axis=1)

    # 初始化StandardScaler并仅对训练集进行拟合
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_scaled)

    # 使用训练集的scaler对测试集进行转换
    X_test_scaled = scaler.transform(X_test_scaled)

    # 将标准化后的特征转为DataFrame，并恢复列名
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.drop(non_scaled_features, axis=1).columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.drop(non_scaled_features, axis=1).columns)

    # 恢复不需要标准化的特征
    X_train_non_scaled = X_train[non_scaled_features].reset_index(drop=True)
    X_test_non_scaled = X_test[non_scaled_features].reset_index(drop=True)

    # 合并标准化后的和未标准化的特征
    X_train_final = pd.concat([X_train_scaled_df, X_train_non_scaled], axis=1)
    X_test_final = pd.concat([X_test_scaled_df, X_test_non_scaled], axis=1)

    # 应用cluster_centroids
    X_train_resampled, y_train_resampled = cluster_centroids.fit_resample(X_train_final, y_train)

    # 训练模型
    model.fit(X_train_resampled, y_train_resampled)

    # 预测
    y_pred = model.predict(X_test_final)

    # 评估性能
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 保存每次的准确率和平衡准确率
    accuracy_scores.append(acc)
    balanced_accuracy_scores.append(balanced_acc)

    # 输出当前折的结果
    print(f"Fold {fold} - Accuracy: {acc:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # 绘制混淆矩阵的热力图
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Bankrupt', 'Bankrupt'],
                yticklabels=['Not Bankrupt', 'Bankrupt'])
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# 计算并输出平均准确率和平均平衡准确率
mean_acc = np.mean(accuracy_scores)
mean_bal_acc = np.mean(balanced_accuracy_scores)

print(f"\nClusterCentroids Average Accuracy: {mean_acc:.4f}")
print(f"ClusterCentroids Average Balanced Accuracy: {mean_bal_acc:.4f}")


