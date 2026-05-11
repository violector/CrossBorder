import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv('results_36000.csv')

# --- 核心修改：合并异常状态到 Reject ---
# 将 M1, M2, M3 视为 Reject 状态进行合并
mapping = {
    'M1': 'Reject',
    'M2': 'Reject',
    'M3': 'Reject'
}
df['final_state'] = df['final_state'].replace(mapping)

# 打印清洗后的状态分布，确保 M1/M2/M3 已消失
print("清洗后的决策状态分布：")
print(df['final_state'].value_counts())
print("-" * 30)


# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# --- Analysis 1: Decision Matrix Heatmap ---
plt.figure(figsize=(10, 6))
decision_matrix = pd.crosstab(df['initial_identified_model'], df['final_state'])
sns.heatmap(decision_matrix, annot=True, cmap="YlGnBu", fmt='d')
plt.title("Decision Distribution: Initial Model vs. Final State")
plt.xlabel("Final State")
plt.ylabel("Initial Model")
plt.show()

# --- Analysis 2: Performance Analysis (Violin + Box Plot) ---
plt.figure(figsize=(12, 7))
sns.violinplot(x='initial_identified_model', y='total_reward', data=df,
               palette="Set2", order=['M1', 'M2', 'M3', 'M4'], inner=None, alpha=0.6)
sns.boxplot(x='initial_identified_model', y='total_reward', data=df,
            width=0.15, color="white", order=['M1', 'M2', 'M3', 'M4'])
plt.title("Reward Probability Density Distribution per Model Type")
plt.xlabel("Initial Model")
plt.ylabel("Total Reward")
plt.show()

# --- Analysis 3: Expanded Correlation Heatmap (Specific Decisions) ---
# We expand 'final_state' into individual columns (One-Hot Encoding)
input_features = ['d', 'b', 'r', 'c']
# Encoding inputs for correlation calculation
df_encoded_inputs = df[input_features].copy()
le = LabelEncoder()
for col in input_features:
    df_encoded_inputs[col] = le.fit_transform(df_encoded_inputs[col].astype(str))

# Encoding final_state into actual outcomes
df_decisions = pd.get_dummies(df['final_state'], prefix='Result')

# Combine encoded inputs, numeric metrics, and expanded decisions
analysis_df = pd.concat([df_encoded_inputs, df_decisions, df[['total_steps', 'total_reward']]], axis=1)

plt.figure(figsize=(14, 10))
# Calculate correlation and slice it to show Features vs. Specific Decisions
corr_matrix = analysis_df.corr()
sub_corr = corr_matrix.loc[input_features, df_decisions.columns]

sns.heatmap(sub_corr, annot=True, cmap="magma", fmt=".2f", linewidths=0.5)
plt.title("Correlation: Input Features vs. Actual Decision Outcomes")
plt.xlabel("Specific Terminal Decisions")
plt.ylabel("Input Features")
plt.show()

# --- Analysis 4: Decision Efficiency ---
plt.figure(figsize=(10, 6))
sns.lineplot(x='total_steps', y='total_reward', hue='initial_identified_model', data=df, marker='o')
plt.title("Impact of Decision Steps on Total Reward")
plt.xlabel("Total Steps")
plt.ylabel("Total Reward")
plt.show()

# --- Analysis 5: Multi-Feature Interaction (Reject Rate) ---
df['is_reject'] = (df['final_state'] == 'Reject').astype(int)
plt.figure(figsize=(12, 8))
interaction_pivot = df.pivot_table(index='d', columns='r', values='is_reject', aggfunc='mean')
sns.heatmap(interaction_pivot, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Interaction: Data Type (d) vs. Risk Flag (r) on Reject Rate")
plt.ylabel("Data Type (d)")
plt.xlabel("High Risk Flag (r)")
plt.show()

# # --- 分析 3: 特征相关性热力图 (Feature Correlation) ---
# df_numeric = df.copy()
# categorical_cols = ['d', 'b', 'r', 'c', 'initial_identified_model', 'final_state']
# for col in categorical_cols:
#     df_numeric[col] = df_numeric[col].astype('category').cat.codes
#
# plt.figure(figsize=(12, 8))
# correlation = df_numeric[['d', 'b', 'r', 'c', 'total_steps', 'total_reward']].corr()
# sns.heatmap(correlation, annot=True, cmap="RdBu_r", center=0)
# plt.title("输入特征与奖励/步数的相关性热力图")
# plt.show()
#
# # --- 分析 4: 决策效率分析 (Efficiency) ---
# # 步数越多，奖励会更高吗？（评估 Agent 是否学会了通过 a_eval 规避风险）
# plt.figure(figsize=(10, 6))
# sns.lineplot(x='total_steps', y='total_reward', hue='initial_identified_model', data=df, marker='o')
# plt.title("决策步数对奖励的影响趋势")
# plt.show()
#
# # --- 打印简单的统计摘要 ---
# print("-" * 30)
# print("策略执行摘要:")
# print(f"平均奖励: {df['total_reward'].mean():.2f}")
# print(f"平均决策步数: {df['total_steps'].mean():.2f}")
# print(f"最常见的终端状态: {df['final_state'].mode()[0]}")
# print("-" * 30)
#
# # 2. 准备相关性分析的数据集
# # 我们关注输入特征 (d, b, r, c, m) 和 结果指标 (total_steps, total_reward, final_state)
# analysis_cols = ['d', 'b', 'r', 'c', 'm', 'total_steps', 'total_reward', 'final_state']
# df_corr = df[analysis_cols].copy()
#
# # 3. 对分类变量进行编码 (Label Encoding)
# # 这样可以将 'multimodal' -> 1, 'structured' -> 2 等，从而计算相关性
# le = LabelEncoder()
# for col in df_corr.columns:
#     if df_corr[col].dtype == 'object':
#         df_corr[col] = le.fit_transform(df_corr[col].astype(str))
#
# # 4. 计算相关系数矩阵
# corr_matrix = df_corr.corr()
#
# # 5. 绘制热力图
# plt.figure(figsize=(12, 10))
#
# # # 使用 RdBu_r 调色板，红色表示正相关，蓝色表示负相关
# # sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, fmt=".2f", linewidths=0.5)
# # "magma" 是一个从黑/紫到红/橙到白色的顺序色图
# sns.heatmap(corr_matrix, annot=True, cmap="magma", fmt=".2f", linewidths=0.5)
#
# plt.title("Correlation Heatmap: Input Features vs. Outcomes", fontsize=15)
# plt.tight_layout()
# plt.show()
#
# # 6. 特别展示：特征对 Reward 的影响排序
# plt.figure(figsize=(8, 6))
# reward_corr = corr_matrix['total_reward'].sort_values(ascending=False).drop('total_reward')
# reward_corr.plot(kind='barh', color='skyblue')
# plt.title("Correlation Strength with Total Reward")
# plt.xlabel("Correlation Coefficient")
# plt.tight_layout()
# plt.show()
#
# input_features = ['d', 'b', 'r', 'c', 'm']
# target_col = 'final_state'
#
# # 复制一份用于分析的数据
# df_analysis = df[input_features + [target_col]].copy()
#
# # 3. 处理输入特征：使用 Label Encoding 将字符串转为数值
# le = LabelEncoder()
# for col in input_features:
#     df_analysis[col] = le.fit_transform(df_analysis[col].astype(str))
#
# # 4. 处理终端决策：使用 One-Hot Encoding
# # 这会把 'final_state' 变成 'final_state_Reject', 'final_state_Accept_M1' 等多列
# df_final_decisions = pd.get_dummies(df_analysis[target_col], prefix='Decision')
# df_encoded = pd.concat([df_analysis[input_features], df_final_decisions], axis=1)
#
# # 5. 计算相关性矩阵
# # 我们只关心“输入特征”列与“决策结果”列之间的交叉相关性
# corr_matrix = df_encoded.corr()
# final_corr = corr_matrix.loc[input_features, df_final_decisions.columns]
#
# # 6. 绘制热力图
# plt.figure(figsize=(12, 8))
#
# # 使用 "magma" 颜色方案，颜色越亮（趋向白色）表示正相关性越强
# sns.heatmap(final_corr, annot=True, cmap="magma", fmt=".2f", linewidths=0.5)
#
# plt.title("Correlation: Input Features vs. Final Decisions", fontsize=15)
# plt.xlabel("Terminal Decisions", fontsize=12)
# plt.ylabel("Input Features", fontsize=12)
# plt.tight_layout()
# plt.show()

# 2. 预处理 m_metrics：将字符串列表拆分为独立列
# ast.literal_eval 能安全地将 "[0.1, 0.2...]" 转为 Python list
metrics_list = df['m_metrics'].apply(ast.literal_eval).tolist()
metrics_df = pd.DataFrame(metrics_list, columns=['m_loss', 'm_accuracy', 'm_recall', 'm_precision'])

# 3. 准备主分析表
# 选择基础特征 + 结果特征
base_features = ['d', 'b', 'r', 'c', 'm']
outcomes = ['total_steps', 'total_reward', 'final_state']
df_analysis = pd.concat([df[base_features], metrics_df, df[outcomes]], axis=1)

# 4. 数值化处理
le = LabelEncoder()
for col in ['d', 'b', 'r', 'c', 'm', 'final_state']:
    df_analysis[col] = le.fit_transform(df_analysis[col].astype(str))

# 5. 计算相关性矩阵
corr_matrix = df_analysis.corr()

# 6. 绘制热力图
plt.figure(figsize=(14, 10))

# 使用 magma 色图，展现从深紫到亮粉的相关性过渡
sns.heatmap(corr_matrix, annot=True, cmap="magma", fmt=".2f", linewidths=0.5)

plt.title("Correlation Heatmap: Input (incl. Metrics) vs. Outcomes", fontsize=15)
plt.tight_layout()
plt.show()

# 7. 针对性分析：最容易导致 Reject 的指标
# 将 final_state 展开为 One-Hot，观察具体指标的影响
if 'final_state' in df.columns:
    df_reject = pd.get_dummies(df['final_state'])['Reject']
    reject_corr = \
    pd.concat([df_analysis.drop(columns=['final_state', 'total_reward', 'total_steps']), df_reject], axis=1).corr()[
        'Reject']

    print("-" * 30)
    print("Metrics correlation with 'Reject' decision:")
    print(reject_corr[['m_loss', 'm_accuracy', 'm_recall', 'm_precision']].sort_values(ascending=False))
    print("-" * 30)

# 设置绘图风格 (英文显示以匹配之前的热力图风格)
sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# 2. 预处理 m_metrics (拆分为独立数值列)
metrics_list = df['m_metrics'].apply(ast.literal_eval).tolist()
metrics_df = pd.DataFrame(metrics_list, columns=['m_loss', 'm_accuracy', 'm_recall', 'm_precision'])

# 3. 准备特征数据 (d, b, r, c, m) 并进行数值化编码
input_features = ['d', 'b', 'r', 'c', 'm']
df_inputs = df[input_features].copy()
le = LabelEncoder()
for col in input_features:
    df_inputs[col] = le.fit_transform(df_inputs[col].astype(str))

# 4. 关键步骤：将 final_state 展开为“真正的结果” (One-Hot Encoding)
# 这会生成 Decision_Accept_M1, Decision_Reject 等列
df_decisions = pd.get_dummies(df['final_state'], prefix='Decision')

# 5. 合并所有维度进行相关性计算
# 包含：输入基础特征 + 拆解后的指标 + 具体的决策结果 + 总奖励
df_full_analysis = pd.concat([df_inputs, metrics_df, df_decisions, df['total_reward'], df['total_steps']], axis=1)

# 6. 提取相关性矩阵
corr_matrix = df_full_analysis.corr()

# --- 绘图 A: 输入特征 + 性能指标 VS 具体决策结果 ---
# 我们只截取 [输入特征+指标] 作为纵轴，[决策结果] 作为横轴
plt.figure(figsize=(16, 10))

# 纵轴选择：基础特征 + 4个指标
y_axis_labels = input_features + ['m_loss', 'm_accuracy', 'm_recall', 'm_precision']
# 横轴选择：所有 Decision_ 开头的列
x_axis_labels = [col for col in df_decisions.columns]

# 提取子矩阵
sub_corr_matrix = corr_matrix.loc[y_axis_labels, x_axis_labels]

sns.heatmap(sub_corr_matrix, annot=True, cmap="magma", fmt=".2f", linewidths=0.5)
plt.title("Correlation: Inputs & Metrics vs. Specific Final Decisions", fontsize=15)
plt.xlabel("Specific Terminal Outcomes", fontsize=12)
plt.ylabel("Input Features & Model Metrics", fontsize=12)
plt.tight_layout()
plt.show()

# --- 绘图 B: 整体全维度热力图 (包含所有变量) ---
plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, cmap="magma", fmt=".2f", linewidths=0.3, annot_kws={"size": 8})
plt.title("Full dimensional Correlation Map", fontsize=16)
plt.tight_layout()
plt.show()

# --- 统计摘要打印 ---
print("-" * 30)
print("Decision Outcome Distribution:")
print(df['final_state'].value_counts())
print("-" * 30)

sns.set_theme(style="whitegrid")

# 2. 预处理：将特征 'd' 展开为具体数值列 (One-Hot)
# 这会生成 d_multimodal, d_structured, d_unstructured_text, d_image 等
df_d_values = pd.get_dummies(df['d'], prefix='d')

# 3. 预处理：将 'final_state' 展开为具体决策列 (One-Hot)
# 这会生成 Decision_Accept_M1, Decision_Reject 等
df_decisions = pd.get_dummies(df['final_state'], prefix='Decision')

# 4. 合并数据进行相关性计算
# 我们只关注 d 的具体值与最终决策之间的关系
df_d_corr = pd.concat([df_d_values, df_decisions], axis=1)

# 5. 计算相关系数矩阵
corr_matrix = df_d_corr.corr()

# 6. 提取子矩阵：纵轴为 d 的不同取值，横轴为最终决策
sub_corr = corr_matrix.loc[df_d_values.columns, df_decisions.columns]

# 7. 绘制热力图
plt.figure(figsize=(14, 8))

# 使用 magma 颜色方案
sns.heatmap(sub_corr, annot=True, cmap="magma", fmt=".2f", linewidths=0.5)

plt.title("Correlation: Specific Values of 'd' vs. Final Decisions", fontsize=15)
plt.xlabel("Terminal Decisions", fontsize=12)
plt.ylabel("Specific Data Types (Values of 'd')", fontsize=12)
plt.tight_layout()
plt.show()

# 8. 辅助分析：打印 d 对应决策的百分比分布
print("-" * 30)
print("Distribution of Decisions per Data Type (d):")
pivot_table = pd.crosstab(df['d'], df['final_state'], normalize='index') * 100
print(pivot_table.round(2).astype(str) + '%')
print("-" * 30)

sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False

# --- Analysis 1: Multi-Feature Interaction (Reject Rate Matrix) ---
# Observation: Interaction between Data Type (d) and Risk Level (r)

df['final_state'] = df['final_state'].replace({'M1': 'Reject', 'M2': 'Reject', 'M3': 'Reject'})
df['is_reject'] = (df['final_state'] == 'Reject').astype(int)

# Split m_metrics into individual columns
metrics_list = df['m_metrics'].apply(ast.literal_eval).tolist()
metrics_df = pd.DataFrame(metrics_list, columns=['m_loss', 'm_accuracy', 'm_recall', 'm_precision'])
df = pd.concat([df, metrics_df], axis=1)
plt.figure(figsize=(12, 8))
interaction_pivot = df.pivot_table(index='d', columns='r', values='is_reject', aggfunc='mean')

sns.heatmap(interaction_pivot, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Reject Rate Interaction: Data Type (d) vs. Risk Flag (r)", fontsize=14)
plt.ylabel("Data Type (d)")
plt.xlabel("High Risk Flag (r)")
plt.tight_layout()
plt.show()

# --- Analysis 2: Reward Distribution (Violin + Box Plot) ---
plt.figure(figsize=(12, 7))
# Violin plot for density
sns.violinplot(x='initial_identified_model', y='total_reward', data=df,
               palette="Set2", order=['M1', 'M2', 'M3', 'M4'], inner=None, alpha=0.6)
# Overlay Box plot for quantiles
sns.boxplot(x='initial_identified_model', y='total_reward', data=df,
            width=0.15, color="white", order=['M1', 'M2', 'M3', 'M4'])
plt.title("Reward Probability Density Distribution per Model Type", fontsize=14)
plt.xlabel("Initial Identified Model")
plt.ylabel("Total Reward")
plt.show()

# --- Analysis 3: Metrics Performance in High-Risk Groups ---
# Example: Comparing 'image' + 'high-risk' group accuracy with global accuracy
high_risk_group = df[(df['d'] == 'image') & (df['r'] == 'high-risk')]
plt.figure(figsize=(10, 6))
sns.histplot(high_risk_group['m_accuracy'], kde=True, color='red', label='High-Risk (Image)', stat="density")
sns.histplot(df['m_accuracy'], kde=True, color='gray', label='Global Baseline', alpha=0.5, stat="density")
plt.title("Accuracy Distribution: High-Risk Group vs. Global Baseline", fontsize=14)
plt.xlabel("Model Accuracy")
plt.ylabel("Density")
plt.legend()
plt.show()

# --- Analysis 4: Decision Steps vs. Final Outcome Heatmap ---
plt.figure(figsize=(10, 6))
step_summary = df.groupby(['initial_identified_model', 'final_state'])['total_steps'].mean().unstack()
sns.heatmap(step_summary, annot=True, cmap="Blues", fmt=".1f")
plt.title("Average Decision Steps per Model and Outcome", fontsize=14)
plt.xlabel("Final Decision State")
plt.ylabel("Initial Model")
plt.show()

# 2. Set Plotting Style (Using Standard Fonts for English)
sns.set_theme(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign displays correctly

# --- Analysis 1: Multi-Feature Interaction (Reject Rate Matrix) ---
plt.figure(figsize=(12, 8))
interaction_pivot = df.pivot_table(index='d', columns='r', values='is_reject', aggfunc='mean')

sns.heatmap(interaction_pivot, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Reject Rate Interaction: Data Type (d) vs. Risk Level (r)")
plt.ylabel("Data Type (d)")
plt.xlabel("Is High Risk (r)")
plt.tight_layout()
plt.show()

# --- Analysis 2: Reward Distribution (Violin + Box Plot) ---
plt.figure(figsize=(12, 7))
# Violin plot for probability density
sns.violinplot(x='initial_identified_model', y='total_reward', data=df,
               palette="Set2", order=['M1', 'M2', 'M3', 'M4'], inner=None, alpha=0.6)
# Narrow boxplot overlaid for quartiles
sns.boxplot(x='initial_identified_model', y='total_reward', data=df,
            width=0.15, color="white", order=['M1', 'M2', 'M3', 'M4'])
plt.title("Reward Probability Density Distribution per Initial Model")
plt.xlabel("Initial Identified Model")
plt.ylabel("Total Reward")
plt.show()

# --- Analysis 3: Specific High-Risk Group Metric Performance ---
# Filter for high-risk combinations (e.g., image + high-risk)
high_risk_group = df[(df['d'] == 'image') & (df['r'] == 'high-risk')]

plt.figure(figsize=(10, 6))
sns.histplot(high_risk_group['m_accuracy'], kde=True, color='red', label='High-Risk Group (Image)', stat="density")
sns.histplot(df['m_accuracy'], kde=True, color='gray', label='Global Baseline (All Data)', alpha=0.5, stat="density")
plt.title("Accuracy Distribution: High-Risk Group vs. Global Baseline")
plt.xlabel("Model Accuracy")
plt.ylabel("Density")
plt.legend()
plt.show()