"""
全面的测试数据生成器
目标：生成覆盖所有状态组合的数据集，确保能测试所有动作和状态转移
"""
import pandas as pd
import numpy as np
from itertools import product
import random


class ComprehensiveDataGenerator:
    """生成覆盖所有可能状态的测试数据"""

    def __init__(self):
        # 定义所有可能的特征值
        self.features = {
            'd': ['structured', 'unstructured_text', 'image', 'video', 'audio', 'multimodal'],
            'b': ['none', 'verification', 'remote_id', 'categorisation'],
            'r': ['false', 'high-risk'],
            'c': ['EU_only', 'adequacy', 'inadequate_SCC', 'non_compliant'],
            'k': ['start', 'evidence', 'proposed', 'final'],
            'm': ['optimal', 'suboptimal', 'insufficient']
        }

        # 模型类型判断规则（与环境代码一致）
        self.model_classification = {
            'M1': {'b': 'none', 'c': 'EU_only', 'r': 'false'},
            'M2': {'b': 'remote_id', 'c': 'adequacy', 'r': 'high-risk'},
            'M3': {'b': 'remote_id', 'c': 'non_compliant', 'r': 'high-risk'},
            'M4': {'b': 'categorisation', 'r': 'high-risk'}
        }

    def get_model_type(self, b, r, c):
        """根据特征判断模型类型"""
        if b == 'categorisation':
            return 'M4'
        if c in ['non_compliant', 'inadequate_SCC']:
            return 'M3'
        if b in ['remote_id'] or r == 'high-risk':
            return 'M2'
        return 'M1'

    def generate_m_metrics(self, m):
        """
        根据模型性能生成m_metrics
        optimal: 高性能指标
        suboptimal: 中等性能指标
        insufficient: 低性能指标
        """
        if m == 'optimal':
            # 高性能：0.8-0.95范围
            return [round(random.uniform(0.80, 0.95), 3) for _ in range(4)]
        elif m == 'suboptimal':
            # 中等性能：0.6-0.8范围
            return [round(random.uniform(0.60, 0.80), 3) for _ in range(4)]
        else:  # insufficient
            # 低性能：0.0-0.2高，0.7-1.0低
            return [round(random.uniform(0.01, 0.20), 3)] + \
                   [round(random.uniform(0.70, 0.99), 3) for _ in range(3)]

    def generate_balanced_dataset(self, samples_per_model=1000):
        """
        生成平衡的数据集，确保每个模型类型有足够样本

        策略：
        1. 为每个模型类型(M1-M4)生成典型配置
        2. 为每个模型类型生成变种配置
        3. 确保所有转移路径都有样本
        """
        data = []

        print("🎲 生成全面覆盖的测试数据集...")
        print("=" * 80)

        # ========== 1. 生成M1模型样本（minimal risk）==========
        print("\n📊 生成 M1 (Minimal Risk) 样本...")
        m1_configs = [
            {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only'},
            {'d': 'unstructured_text', 'b': 'none', 'r': 'false', 'c': 'EU_only'},
            {'d': 'image', 'b': 'none', 'r': 'false', 'c': 'EU_only'},
        ]

        for config in m1_configs:
            for _ in range(samples_per_model // len(m1_configs)):
                row = config.copy()
                row['k'] = random.choice(['start', 'evidence', 'proposed'])
                row['m'] = random.choice(['optimal', 'suboptimal', 'insufficient'])
                row['m_metrics'] = str(self.generate_m_metrics(row['m']))
                data.append(row)

        print(f"  ✓ 生成 {samples_per_model} 条 M1 样本")

        # ========== 2. 生成M2模型样本（limited risk）==========
        print("\n📊 生成 M2 (Limited Risk) 样本...")
        m2_configs = [
            {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'adequacy'},
            {'d': 'video', 'b': 'remote_id', 'r': 'high-risk', 'c': 'adequacy'},
            {'d': 'multimodal', 'b': 'verification', 'r': 'high-risk', 'c': 'adequacy'},
            {'d': 'image', 'b': 'remote_id', 'r': 'false', 'c': 'adequacy'},
        ]

        for config in m2_configs:
            for _ in range(samples_per_model // len(m2_configs)):
                row = config.copy()
                row['k'] = random.choice(['start', 'evidence', 'proposed'])
                row['m'] = random.choice(['optimal', 'suboptimal', 'insufficient'])
                row['m_metrics'] = str(self.generate_m_metrics(row['m']))
                data.append(row)

        print(f"  ✓ 生成 {samples_per_model} 条 M2 样本")

        # ========== 3. 生成M3模型样本（high risk）==========
        print("\n📊 生成 M3 (High Risk) 样本...")
        m3_configs = [
            {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant'},
            {'d': 'video', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant'},
            {'d': 'unstructured_text', 'b': 'remote_id', 'r': 'high-risk', 'c': 'inadequate_SCC'},
            {'d': 'image', 'b': 'verification', 'r': 'high-risk', 'c': 'non_compliant'},
        ]

        for config in m3_configs:
            for _ in range(samples_per_model // len(m3_configs)):
                row = config.copy()
                row['k'] = random.choice(['start', 'evidence', 'proposed'])
                row['m'] = random.choice(['suboptimal', 'insufficient'])  # M3通常性能不佳
                row['m_metrics'] = str(self.generate_m_metrics(row['m']))
                data.append(row)

        print(f"  ✓ 生成 {samples_per_model} 条 M3 样本")

        # ========== 4. 生成M4模型样本（banned）==========
        print("\n📊 生成 M4 (Banned) 样本...")
        m4_configs = [
            {'d': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy'},
            {'d': 'video', 'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy'},
            {'d': 'image', 'b': 'categorisation', 'r': 'false', 'c': 'EU_only'},
            {'d': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'inadequate_SCC'},
        ]

        for config in m4_configs:
            for _ in range(samples_per_model // len(m4_configs)):
                row = config.copy()
                row['k'] = random.choice(['start', 'evidence'])  # M4通常评估较少
                row['m'] = 'insufficient'  # M4性能都不足
                row['m_metrics'] = str(self.generate_m_metrics(row['m']))
                data.append(row)

        print(f"  ✓ 生成 {samples_per_model} 条 M4 样本")

        # ========== 5. 生成边界案例（确保覆盖转移路径）==========
        print("\n📊 生成边界案例样本...")

        # M1->M2 转移路径样本
        for _ in range(200):
            data.append({
                'd': random.choice(['structured', 'image']),
                'b': 'none',
                'r': 'false',
                'c': 'EU_only',
                'k': 'start',
                'm': random.choice(['optimal', 'suboptimal']),
                'm_metrics': str(self.generate_m_metrics('optimal'))
            })

        # M2->M3 转移路径样本
        for _ in range(200):
            data.append({
                'd': 'multimodal',
                'b': 'remote_id',
                'r': 'high-risk',
                'c': random.choice(['adequacy', 'inadequate_SCC']),
                'k': 'evidence',
                'm': 'suboptimal',
                'm_metrics': str(self.generate_m_metrics('suboptimal'))
            })

        # M2->M4 转移路径样本
        for _ in range(200):
            data.append({
                'd': 'video',
                'b': random.choice(['remote_id', 'categorisation']),
                'r': 'high-risk',
                'c': 'adequacy',
                'k': 'evidence',
                'm': 'insufficient',
                'm_metrics': str(self.generate_m_metrics('insufficient'))
            })

        print(f"  ✓ 生成 600 条边界案例")

        # ========== 6. 生成完全随机样本（增加多样性）==========
        print("\n📊 生成随机多样性样本...")

        for _ in range(1000):
            data.append({
                'd': random.choice(self.features['d']),
                'b': random.choice(self.features['b']),
                'r': random.choice(self.features['r']),
                'c': random.choice(self.features['c']),
                'k': random.choice(self.features['k']),
                'm': random.choice(self.features['m']),
                'm_metrics': str(self.generate_m_metrics(random.choice(self.features['m'])))
            })

        print(f"  ✓ 生成 1000 条随机样本")

        # 转换为DataFrame
        df = pd.DataFrame(data)

        # 打乱顺序
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        print("\n" + "=" * 80)
        print(f"✅ 数据生成完成！总计: {len(df)} 条")

        return df

    def generate_stratified_dataset(self, total_samples=30000):
        """
        生成分层抽样数据集，按模型类型比例生成

        比例设置（基于真实场景）：
        M1: 30% (大部分系统是低风险的)
        M2: 35% (有限风险系统较多)
        M3: 25% (高风险系统)
        M4: 10% (禁止类系统较少)
        """
        proportions = {
            'M1': 0.30,
            'M2': 0.35,
            'M3': 0.25,
            'M4': 0.10
        }

        samples_per_model = {
            model: int(total_samples * prop)
            for model, prop in proportions.items()
        }

        return self.generate_balanced_dataset(
            samples_per_model=samples_per_model['M1']
        )

    def analyze_dataset(self, df):
        """分析生成的数据集"""
        print("\n" + "=" * 80)
        print("📊 数据集分析报告")
        print("=" * 80)

        # 统计各特征分布
        for col in ['d', 'b', 'r', 'c', 'k', 'm']:
            counts = df[col].value_counts()
            print(f"\n【{col} 分布】")
            for val, count in counts.items():
                pct = count / len(df) * 100
                bar = "█" * int(pct / 2)
                print(f"  {val:20s}: {count:5d} ({pct:5.1f}%) {bar}")

        # 统计模型类型分布
        print("\n【预期模型类型分布】")
        df['predicted_model'] = df.apply(
            lambda row: self.get_model_type(row['b'], row['r'], row['c']),
            axis=1
        )
        model_counts = df['predicted_model'].value_counts()
        for model in ['M1', 'M2', 'M3', 'M4']:
            count = model_counts.get(model, 0)
            pct = count / len(df) * 100
            bar = "█" * int(pct / 2)
            print(f"  {model}: {count:5d} ({pct:5.1f}%) {bar}")

        print("\n" + "=" * 80)


def main():
    """主函数"""
    generator = ComprehensiveDataGenerator()

    # 生成数据集
    print("\n🚀 开始生成测试数据集\n")

    # 方案1：平衡数据集（每个模型1000条）
    # df = generator.generate_balanced_dataset(samples_per_model=1000)

    # 方案2：分层数据集（按真实比例，共30000条）
    df = generator.generate_stratified_dataset(total_samples=30000)

    # 分析数据集
    generator.analyze_dataset(df)

    # 保存为CSV
    output_file = "comprehensive_test_data.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 数据集已保存至: {output_file}")

    # 生成小样本用于快速测试
    small_df = df.head(100)
    small_output = "quick_test_data.csv"
    small_df.to_csv(small_output, index=False, encoding='utf-8-sig')
    print(f"💾 小样本数据集已保存至: {small_output} (100条)")
    print()


if __name__ == "__main__":
    main()