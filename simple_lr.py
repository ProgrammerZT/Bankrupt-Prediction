import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

# 假设数据集为 'data_transformed.csv'
data = pd.read_csv('../data_transformed1.csv')

# 特征和目标变量
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# 列出不需要标准化的特征
non_scaled_features = ['Net Income Flag', 'Liability-Assets Flag']

# 划分数据集为训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# 使用逻辑回归训练模型
log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
log_reg_model.fit(X_train_final, y_train)

# 进行预测
y_pred = log_reg_model.predict(X_test_final)  # 使用测试集进行预测

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 输出混淆矩阵带标签
print("Confusion Matrix for Logistic Regression:")
print(pd.DataFrame(cm, index=['Negative (0)', 'Positive (1)'],
                   columns=['Predicted Negative (0)', 'Predicted Positive (1)']))

# 计算准确率和平衡准确率
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

# 输出逻辑回归的准确率和平衡准确率
print(f"Logistic Regression Accuracy: {accuracy:.4f} Balanced Accuracy: {balanced_accuracy:.4f}")

# 创建一个DummyClassifier，始终预测'0' (即总是预测为Negative)
dummy_clf = DummyClassifier(strategy="constant", constant=0, random_state=42)
dummy_clf.fit(X_train_final, y_train)

# 进行预测
dummy_pred = dummy_clf.predict(X_test_final)

# 计算DummyClassifier的准确率和平衡准确率
dummy_accuracy = accuracy_score(y_test, dummy_pred)
dummy_balanced_accuracy = balanced_accuracy_score(y_test, dummy_pred)

# 输出DummyClassifier的准确率和平衡准确率
print(
    f"Dummy Classifier (Always Predicts Negative) - Accuracy: {dummy_accuracy:.4f} Balanced Accuracy: {dummy_balanced_accuracy:.4f}")
