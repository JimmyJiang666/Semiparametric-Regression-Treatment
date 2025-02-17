# 半参数处理效应交互模型

[English](README.md)

## 项目概述

本项目实现了一个创新的半参数回归模型，用于分析连续处理变量与协变量之间的交互作用。该模型特别适用于医疗研究、临床试验和精准医疗等领域，能够有效地估计和评估个性化治疗效果。

## 主要特点

- 🔬 **创新的统计方法**：实现了重复 Nadaraya-Watson 回归估计器，能够处理连续型处理变量
- 📊 **灵活的建模框架**：支持多种核函数和带宽选择方法
- 🛠️ **优化算法集成**：整合了多种优化方法（Hyperopt、CMA-ES、差分进化等）
- 🔍 **交叉验证**：实现了完整的交叉验证框架，用于模型选择和超参数调优
- 📈 **可视化工具**：提供丰富的可视化功能，包括热力图、ROC曲线等

## 项目结构

```
📦 semiparametric-treatment-interaction
 ┣ 📂 semiparametric_treatment_interaction/  # 主包目录
 ┃ ┣ 📜 model.py            # 核心模型训练与预测
 ┃ ┣ 📜 kernels.py          # 核函数实现
 ┃ ┣ 📜 objectives.py       # 优化目标函数
 ┃ ┣ 📜 optimizers.py       # 优化算法
 ┃ ┣ 📜 utils.py            # 工具函数与数据处理
 ┃ ┣ 📜 visualization.py    # 可视化工具
 ┃ ┗ 📜 __init__.py         # 包初始化
 ┣ 📂 examples/             # 示例笔记本
 ┃ ┣ 📜 simulation.ipynb    # 模拟数据分析
 ┃ ┣ 📜 beta_xi_conf.ipynb  # 参数置信区间估计
 ┃ ┗ 📜 diag_score_comparison.ipynb  # 诊断评分比较
 ┣ 📂 tests/                # 测试套件
 ┣ 📂 docs/                 # 文档
 ┣ 📂 figures/              # 图片
 ┣ 📜 setup.py             # 包安装配置
 ┣ 📜 requirements.txt     # 项目依赖
 ┣ 📜 README.md            # 英文文档
 ┣ 📜 README_CN.md         # 中文文档
 ┗ 📜 LICENSE             # MIT 许可证 

## 核心模块

1. **模型训练** (`model.py`)
   - 交叉验证
   - 模型拟合
   - 预测函数

2. **核函数** (`kernels.py`)
   - Nadaraya-Watson 估计器
   - 高维核平滑
   - 数值稳定实现

3. **优化方法** (`optimizers.py`)
   - Hyperopt 优化
   - CMA-ES 算法
   - 差分进化
   - Optuna 框架

4. **目标函数** (`objectives.py`)
   - 基础目标函数
   - Lasso 正则化目标函数
   - 专用优化器目标函数

5. **工具函数** (`utils.py`)
   - 数据处理
   - 评估指标
   - 辅助函数

6. **可视化** (`visualization.py`)
   - 热力图生成
   - ROC 曲线
   - 分布图
   - 3D 表面图

## 应用场景

- 🏥 临床试验分析
- 💊 个性化医疗研究
- 📊 生物统计学研究
- 🔬 医学研究数据分析
- 📈 连续处理效应评估

## 技术要求

- Python 3.8+
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 0.24.0
- 完整依赖列表请参见 `requirements.txt`

## 安装方法

1. 通过 PyPI 安装（即将推出）：
```bash
pip install semiparametric-treatment-interaction
```

2. 从源代码安装：
```bash
# 克隆仓库
git clone https://github.com/yourusername/semiparametric-treatment-interaction.git
cd Semiparametric-Treatment-Interaction-main

# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装包
pip install -e .
```

## 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@article{your-paper-reference,
  title={Learning Interactions Between Continuous Treatments and Covariates with a Semiparametric Model},
  author={Your Name},
  journal={Conference on Health, Inference, and Learning (CHIL)},
  year={2025}
}
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目维护者：[Muyan Jiang]
- 邮箱：[muyan_jiang@berkeley.edu]

