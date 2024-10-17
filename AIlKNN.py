import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import AllKNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

def select_best_k_and_plot_confusion_matrix(data_path, non_scaled_features, k_neighbors_list):
    """
    根据给定的数据路径、非标准化特征以及 AllKNN 的 k_neighbors 值列表，
    选择最佳的 k 值，并绘制每个折叠的混淆矩阵，同时根据模型结果按重要性排序特征。

    Parameters:
    - data_path (str): 数据集路径。
    - non_scaled_features (list): 不需要标准化的特征列表。
    - k_neighbors_list (list): 需要尝试的 k_neighbors 值列表。
    """
    # 读取数据
    data = pd.read_csv(data_path)

    # 特征和目标变量
    X = data.drop('Bankrupt?', axis=1)
    y = data['Bankrupt?']

    # 初始化 KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results_allknn = []

    # 对每个 k_neighbors 值进行交叉验证 (AllKNN)
    for k in k_neighbors_list:
        fold_balanced_accuracies_allknn = []

        for train_index, test_index in kf.split(X):
            # 划分训练集和测试集
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 在训练集上标准化特征
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.drop(non_scaled_features, axis=1))

            # 使用相同的缩放器转换测试集
            X_test_scaled = scaler.transform(X_test.drop(non_scaled_features, axis=1))

            # 将标准化后的特征转为 DataFrame，并恢复列名
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.drop(non_scaled_features, axis=1).columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.drop(non_scaled_features, axis=1).columns)

            # 恢复不需要标准化的特征
            X_train_final = pd.concat([X_train_scaled_df, X_train[non_scaled_features].reset_index(drop=True)], axis=1)
            X_test_final = pd.concat([X_test_scaled_df, X_test[non_scaled_features].reset_index(drop=True)], axis=1)

            # 使用 AllKNN 进行下采样
            allknn = AllKNN(n_neighbors=k)
            X_resampled_allknn, y_resampled_allknn = allknn.fit_resample(X_train_final, y_train)

            # 训练模型
            model = LogisticRegression(max_iter=1000)
            model.fit(X_resampled_allknn, y_resampled_allknn)

            # 预测
            y_pred_allknn = model.predict(X_test_final)

            # 计算平衡准确率
            balanced_acc_allknn = balanced_accuracy_score(y_test, y_pred_allknn)
            fold_balanced_accuracies_allknn.append(balanced_acc_allknn)

        # 计算当前 k_neighbors 的平均平衡准确率
        mean_balanced_acc_allknn = np.mean(fold_balanced_accuracies_allknn)
        results_allknn.append((k, mean_balanced_acc_allknn))

    # 找到最佳的 k_neighbors 值
    best_k, best_balanced_acc = max(results_allknn, key=lambda x: x[1])
    print(f"\n最佳 k_neighbors (AllKNN): {best_k}，平均平衡准确率: {best_balanced_acc:.4f}")

    # 使用最佳 k_neighbors 进行交叉验证并绘制混淆矩阵
    fold_idx = 1
    feature_importance = np.zeros(X.shape[1])  # 用于存储特征重要性的累加
    feature_names = X.columns

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 在训练集上标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.drop(non_scaled_features, axis=1))
        X_test_scaled = scaler.transform(X_test.drop(non_scaled_features, axis=1))

        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.drop(non_scaled_features, axis=1).columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.drop(non_scaled_features, axis=1).columns)

        X_train_final = pd.concat([X_train_scaled_df, X_train[non_scaled_features].reset_index(drop=True)], axis=1)
        X_test_final = pd.concat([X_test_scaled_df, X_test[non_scaled_features].reset_index(drop=True)], axis=1)

        # 使用 AllKNN 进行下采样
        allknn = AllKNN(n_neighbors=best_k)
        X_resampled_allknn, y_resampled_allknn = allknn.fit_resample(X_train_final, y_train)

        # 训练模型并进行预测
        model = LogisticRegression(max_iter=1000)
        model.fit(X_resampled_allknn, y_resampled_allknn)
        y_pred_allknn = model.predict(X_test_final)

        # 计算混淆矩阵
        conf_matrix_allknn = confusion_matrix(y_test, y_pred_allknn)

        print(f"\nFold {fold_idx} Classification Report:")
        print(classification_report(y_true=y_test, y_pred=y_pred_allknn))

        # 绘制混淆矩阵
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix_allknn, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Bankrupt', 'Bankrupt'],
                    yticklabels=['Not Bankrupt', 'Bankrupt'])
        plt.title(f'Confusion Matrix for Fold {fold_idx} (k={best_k})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # 累加特征重要性
        feature_importance += np.abs(model.coef_[0])

        fold_idx += 1

    # 计算平均特征重要性
    feature_importance /= kf.get_n_splits()

    # 创建特征重要性的 DataFrame 并按重要性排序
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # 打印特征重要性表格
    print("\n特征重要性排序：")
    print(tabulate(feature_importance_df, headers='keys', tablefmt='pretty', showindex=False))

    # 生成特征重要性条形图
    plt.figure(figsize=(9,25))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Feature')
    plt.grid(axis='x')
    plt.tight_layout()

    # 保存特征重要性图
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()

    return feature_importance_df

# 使用函数
feature_importance_df = select_best_k_and_plot_confusion_matrix(
    data_path='../data_transformed1.csv',
    non_scaled_features=['Net Income Flag', 'Liability-Assets Flag'],
    k_neighbors_list=[43, 41, 39, 37, 35, 33, 31, 29, 27]
)
