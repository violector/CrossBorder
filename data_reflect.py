import pandas as pd
import ast
import numpy as np


def map_m_performance(l, a, r, p):
    """
    基于 Loss, Accuracy, Recall, Precision 的综合性能映射
    规则：满足 0.9/0.05 为 optimal，0.8/0.15 为 suboptimal，其余为 insufficient
    """
    score_list = []

    # Loss 评分 (越低越好)
    if l < 0.05:
        score_list.append(0)
    elif l < 0.15:
        score_list.append(1)
    else:
        score_list.append(2)

    # Accuracy/Recall/Precision 评分 (越高越好)
    for val in [a, r, p]:
        if val > 0.90:
            score_list.append(0)
        elif val > 0.80:
            score_list.append(1)
        else:
            score_list.append(2)

    # 综合判定：采用最差指标原则 (Bottleneck Principle)
    max_score = max(score_list)
    mapping = {0: 'optimal', 1: 'suboptimal', 2: 'insufficient'}
    return mapping[max_score]


def transform_csv(input_file, output_file):
    # 读取原始数据
    df = pd.read_csv(input_file)

    new_rows = []

    for _, row in df.iterrows():
        # --- 1. 映射 d (Dataset Type) ---
        domain = str(row['domain']).lower().strip()
        intent = str(row['intent']).lower().strip()

        # 优先级逻辑：先看 domain，再看 intent
        if domain == 'natural language processing':
            d_val = 'unstructured_text'
        elif intent == 'traffic prediction':
            d_val = 'video'
        elif domain == 'forecasting':
            d_val = 'structured'
        elif intent in ['image classification', 'segmentation']:
            d_val = 'image'
        elif intent in ['object detection', 'cost minimization', 'resource allocation']:
            d_val = 'multimodal'
        elif intent in ['pattern recognition', 'route optimization']:
            d_val = 'unstructured_text'
        elif intent == 'imputation':
            d_val = 'structured'
        else:
            d_val = 'structured'  # 默认兜底

        # --- 2. 映射 b (Biometric Category) ---
        field = str(row['field']).lower().strip()
        b_map = {
            'healthcare': 'verification',
            'manufacturing': 'remote_id',
            'energy': 'remote_id',
            'transportation': 'categorisation',
            'logistics': 'none'
        }
        b_val = b_map.get(field, 'none')

        # --- 3. 映射 r (Risk Indicators) - 已更新规则 ---
        # 除了 logistics 和 transportation 是 false，其他都是 high-risk
        if field in ['logistics', 'transportation']:
            r_val = 'false'
        else:
            r_val = 'high-risk'

        # --- 4. 映射 c (Cross-border / Nodes Dimension) ---
        try:
            # 将字符串形式的列表 "[64, 64]" 转换为真正的 list 并算长度
            nodes_list = ast.literal_eval(row['nodes'])
            dim = len(nodes_list)
            if dim == 3:
                c_val = 'EU_only'
            elif dim == 4:
                c_val = 'adequacy'
            elif dim == 5:
                c_val = 'inadequate_SCC'
            elif dim >= 6:
                c_val = 'non_compliant'
            else:
                c_val = 'EU_only'
        except:
            c_val = 'EU_only'

        # --- 5. 映射 k (Classification Stage) ---
        k_val = 'start'

        # --- 6. 映射 m & m_metrics ---
        loss = float(row['loss'])
        acc = float(row['accuracy'])
        rec = float(row['recall'])
        pre = float(row['precision'])

        m_val = map_m_performance(loss, acc, rec, pre)
        # 将四个指标存储为列表格式
        m_metrics = [loss, acc, rec, pre]

        # 构造新行
        new_rows.append({
            'd': d_val,
            'b': b_val,
            'r': r_val,
            'c': c_val,
            'k': k_val,
            'm': m_val,
            'm_metrics': m_metrics
        })

    # 导出新 CSV
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_file, index=False)
    print(f"✅ 转换完成！结果已写入: {output_file}")
    print(f"📊 总计转换条数: {len(new_df)}")


if __name__ == "__main__":
    # 执行脚本，请确保 data.csv 在同一目录下
    transform_csv('data.csv', 'transformed_data.csv')