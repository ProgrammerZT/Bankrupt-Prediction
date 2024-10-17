from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
import numpy as np
import pandas as pd

# 假设数据集为 'data_transformed1.csv'
data = pd.read_csv('../data_transformed1.csv')

# 特征和目标变量
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# 列出不需要标准化的特征
non_scaled_features = ['Net Income Flag', 'Liability-Assets Flag']

# 确定哪些特征是类别型的
categorical_features = [X.columns.get_loc(col) for col in non_scaled_features]

# 初始化 KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 设置要尝试的 k_neighbors 值（SMOTENC）和 n_neighbors 值（ENN）
k_neighbors_list = [41, 39, 37, 35, 33, 31, 30, 29, 28, 27]
enn_neighbors_list = range(1, 11)  # ENN的n_neighbors从0到10
results = []

# 对每个 k_neighbors 和 n_neighbors 值进行交叉验证
for k_smote in k_neighbors_list:
    for k_enn in enn_neighbors_list:
        fold_accuracies = []
        fold_balanced_accuracies = []

        for train_index, test_index in kf.split(X):
            # 划分训练集和测试集
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 在训练集上标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.drop(non_scaled_features, axis=1))
            X_test_scaled = scaler.transform(X_test.drop(non_scaled_features, axis=1))

            # 将标准化后的特征转为 DataFrame，并恢复列名
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.drop(non_scaled_features, axis=1).columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.drop(non_scaled_features, axis=1).columns)

            # 恢复不需要标准化的特征
            X_train_final = pd.concat([X_train_scaled_df, X_train[non_scaled_features].reset_index(drop=True)], axis=1)
            X_test_final = pd.concat([X_test_scaled_df, X_test[non_scaled_features].reset_index(drop=True)], axis=1)

            # 使用 SMOTE-NC 进行过采样，随后使用 SMOTEENN 进行下采样
            smote_nc = SMOTENC(categorical_features=categorical_features, k_neighbors=k_smote, random_state=42)
            enn = EditedNearestNeighbours(n_neighbors=k_enn)
            smote_enn = SMOTEENN(smote=smote_nc, enn=enn, random_state=42)

            # 进行过采样和下采样
            X_resampled, y_resampled = smote_enn.fit_resample(X_train_final, y_train)

            # 训练模型
            model = LogisticRegression(max_iter=1000)
            model.fit(X_resampled, y_resampled)

            # 预测
            y_pred = model.predict(X_test_final)

            # 计算准确性和平衡准确率
            acc = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)

            fold_accuracies.append(acc)
            fold_balanced_accuracies.append(balanced_acc)

        # 计算当前 k_neighbors 和 n_neighbors 的平均准确率和平衡准确率
        mean_acc = np.mean(fold_accuracies)
        mean_balanced_acc = np.mean(fold_balanced_accuracies)
        results.append((k_smote, k_enn, mean_acc, mean_balanced_acc))

        print(
            f"k_neighbors: {k_smote}, n_neighbors: {k_enn}, Average Accuracy: {mean_acc:.4f}, Average Balanced Accuracy: {mean_balanced_acc:.4f}")

# 找到最佳的 k_neighbors 和 n_neighbors
best_result = max(results, key=lambda x: x[3])  # 根据均衡准确率查找最佳结果
best_k_smote, best_k_enn, best_acc, best_balanced_acc = best_result

print(
    f"\nBest k_neighbors: {best_k_smote}, Best n_neighbors: {best_k_enn} with Average Accuracy: {best_acc:.4f} and Average Balanced Accuracy: {best_balanced_acc:.4f}")

