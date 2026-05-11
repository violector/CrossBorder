import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. 数据加载与基础配置
# ==========================================
df = pd.read_csv('results_36000.csv')

# 设置全局绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 2. 核心预处理 (统一处理，避免重复)
# ==========================================

# A. 状态合并与清洗
mapping = {'M1': 'Reject', 'M2': 'Reject', 'M3': 'Reject'}
df['final_state'] = df['final_state'].replace(mapping)
df['is_reject'] = (df['final_state'] == 'Reject').astype(int)

# B. 解析 m_metrics 字符串列表
metrics_list = df['m_metrics'].apply(ast.literal_eval).tolist()
metrics_cols = ['m_loss', 'm_accuracy', 'm_recall', 'm_precision']
df_metrics = pd.DataFrame(metrics_list, columns=metrics_cols, index=df.index)
df = pd.concat([df, df_metrics], axis=1)

# C. 定义输入变量
input_features = ['d', 'b', 'r', 'c', 'm']


# ==========================================
# 3. 统计分析函数定义
# ==========================================

def plot_decision_matrix(df):
    """分析初始模型与最终决策的对应关系"""
    plt.figure(figsize=(10, 6))
    decision_matrix = pd.crosstab(df['initial_identified_model'], df['final_state'])
    sns.heatmap(decision_matrix, annot=True, cmap="YlGnBu", fmt='d')
    plt.title("Decision Matrix: Initial Model vs Final State")
    plt.show()


def plot_reward_distribution(df):
    """分析不同初始模型的奖励分布"""
    plt.figure(figsize=(12, 7))
    order = ['M1', 'M2', 'M3', 'M4']
    sns.violinplot(x='initial_identified_model', y='total_reward', data=df,
                   palette="Set2", order=order, inner=None, alpha=0.6)
    sns.boxplot(x='initial_identified_model', y='total_reward', data=df,
                width=0.15, color="white", order=order)
    plt.title("Total Reward Distribution per Initial Model")
    plt.show()


def plot_correlation_heatmap(df, input_features):
    """分析输入特征、指标与决策结果的相关性"""
    # 编码用于计算
    df_corr = df.copy()
    le = LabelEncoder()
    for col in input_features + ['final_state']:
        df_corr[col] = le.fit_transform(df_corr[col].astype(str))

    plt.figure(figsize=(12, 10))
    # 计算全维度相关性
    cols_to_corr = input_features + ['m_loss', 'm_accuracy', 'm_recall', 'm_precision', 'total_reward', 'final_state']
    corr_matrix = df_corr[cols_to_corr].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="magma", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap: Inputs, Metrics & Outcomes")
    plt.show()


def plot_reject_rate_interaction(df):
    """分析 d 和 r 对拒绝率的交互影响"""
    plt.figure(figsize=(10, 7))
    pivot = df.pivot_table(index='d', columns='r', values='is_reject', aggfunc='mean')
    sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title("Reject Rate Interaction: Data Type (d) vs Risk Level (r)")
    plt.show()


def plot_accuracy_analysis(df, features):
    """
    深度分析：针对每一个输入变量，分析其对 m_accuracy 的影响分布
    """
    for col in features:
        plt.figure(figsize=(12, 6))
        # 使用 Boxenplot 适合展示大规模数据的分布，比 Boxplot 更细致
        sns.boxenplot(x=col, y='m_accuracy', data=df, palette="viridis")
        # 叠加一个条形图显示均值线
        plt.title(f"Model Accuracy Distribution by Feature: {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_steps_vs_reward(df):
    """决策步数对奖励的影响"""
    plt.figure(figsize=(10, 6))
    # 将 errorbar=None 改为 ci=None 以兼容旧版本 Seaborn
    sns.lineplot(x='total_steps', y='total_reward', hue='initial_identified_model',
                 data=df, marker='o', ci=None)
    plt.title("Impact of Decision Steps on Total Reward")
    plt.show()


# ==========================================
# 4. 执行流程
# ==========================================

# 1. 基本决策与奖励分析
plot_decision_matrix(df)
plot_reward_distribution(df)
plot_steps_vs_reward(df)

# 2. 交互与相关性分析
plot_reject_rate_interaction(df)
plot_correlation_heatmap(df, input_features)

# 3. 针对所有输入变量的 Accuracy 专项分析
print("正在生成各输入变量对 Accuracy 的影响分析图...")
plot_accuracy_analysis(df, input_features)

# 4. 打印统计摘要
print("\n" + "=" * 40)
print("策略执行汇总报告")
print("=" * 40)
summary = {
    "平均奖励": df['total_reward'].mean(),
    "平均步数": df['total_steps'].mean(),
    "拒绝总数": df['is_reject'].sum(),
    "整体平均准确率": df['m_accuracy'].mean()
}
for k, v in summary.items():
    print(f"{k}: {v:.4f}")

print("\n各数据类型(d)下的平均准确率与拒绝率:")
type_summary = df.groupby('d').agg({
    'm_accuracy': 'mean',
    'is_reject': 'mean'
}).rename(columns={'is_reject': 'reject_rate'})
print(type_summary.round(4))
print("=" * 40)