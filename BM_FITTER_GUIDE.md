# BirchMurnaghanFitter 调用格式速查

## 📋 基本调用格式

### ⭐ 最简单的调用方式（推荐）

```python
from birch_murnaghan_fitter import BirchMurnaghanFitter

# 创建拟合器
fitter = BirchMurnaghanFitter()

# 一行代码完成完整分析
results = fitter.analyze(
    original_csv='data/original_phase.csv',
    new_csv='data/new_phase.csv',
    output_dir='output/BM_fitting'
)
```

---

## 📦 初始化参数

```python
fitter = BirchMurnaghanFitter(
    V0_bounds=(0.8, 1.3),         # V0范围：(min, max) × max_volume
    B0_bounds=(50, 500),          # B0范围：(min, max) GPa
    B0_prime_bounds=(2.5, 6.5),   # B0'范围：(min, max)
    max_iterations=10000          # 最大迭代次数
)
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `V0_bounds` | `(0.8, 1.3)` | V0边界，作为最大实验体积的倍数 |
| `B0_bounds` | `(50, 500)` | 体积模量B0的边界范围（GPa）|
| `B0_prime_bounds` | `(2.5, 6.5)` | B0'的边界范围（无量纲）|
| `max_iterations` | `10000` | curve_fit最大迭代次数 |

---

## 🎯 主要方法

### 1️⃣ `analyze()` - 完整分析流程

```python
results = fitter.analyze(
    original_csv='path/to/original.csv',
    new_csv='path/to/new.csv',
    output_dir='output/directory'  # 可选，不指定则不保存
)
```

**返回值**:
```python
{
    'original_phase': {
        '2nd_order': {
            'V0': ..., 'V0_err': ...,
            'B0': ..., 'B0_err': ...,
            'B0_prime': 4.0, 'B0_prime_err': 0,
            'R_squared': ..., 'RMSE': ...,
            'fitted_P': [...]
        },
        '3rd_order': {...}
    },
    'new_phase': {...}
}
```

### 2️⃣ `load_data_from_csv()` - 从CSV加载数据

```python
success = fitter.load_data_from_csv(
    'data/original_phase.csv',
    'data/new_phase.csv'
)
```

### 3️⃣ `set_data_manually()` - 手动设置数据

```python
import numpy as np

V_orig = np.array([16.8, 16.5, 16.2, 15.9])
P_orig = np.array([0.0, 5.0, 10.0, 15.0])
V_new = np.array([15.5, 15.2, 14.9])
P_new = np.array([15.0, 20.0, 25.0])

fitter.set_data_manually(V_orig, P_orig, V_new, P_new)
```

### 4️⃣ `fit_all_phases()` - 拟合所有相

```python
results_orig, results_new = fitter.fit_all_phases()
```

### 5️⃣ `fit_single_phase()` - 拟合单个相

```python
results = fitter.fit_single_phase(
    V_data=np.array([16.8, 16.5, 16.2]),
    P_data=np.array([0.0, 5.0, 10.0]),
    phase_name="Test Phase"
)
```

### 6️⃣ `plot_pv_curves()` - 绘制P-V曲线

```python
fitter.plot_pv_curves(save_path='output/pv_curves.png')  # 保存
fitter.plot_pv_curves()  # 只显示，不保存
```

### 7️⃣ `plot_residuals()` - 绘制残差图

```python
fitter.plot_residuals(save_path='output/residuals.png')
```

### 8️⃣ `save_results_to_csv()` - 保存结果

```python
df = fitter.save_results_to_csv('output/results.csv')
```

---

## 🔧 静态方法（工具函数）

不需要创建实例即可使用:

```python
from birch_murnaghan_fitter import BirchMurnaghanFitter

# 2阶BM方程计算压力
P = BirchMurnaghanFitter.birch_murnaghan_2nd(
    V=15.5,      # 体积 (Å³/atom)
    V0=16.8,     # 零压体积 (Å³/atom)
    B0=150       # 体积模量 (GPa)
)

# 3阶BM方程计算压力
P = BirchMurnaghanFitter.birch_murnaghan_3rd(
    V=15.5,
    V0=16.8,
    B0=150,
    B0_prime=4.0
)
```

---

## 💡 常用调用示例

### 示例 1: 完整分析（最简单）

```python
from birch_murnaghan_fitter import BirchMurnaghanFitter

fitter = BirchMurnaghanFitter()
results = fitter.analyze(
    'data/original_phase.csv',
    'data/new_phase.csv',
    'output/BM_fitting'
)
```

### 示例 2: 自定义参数

```python
fitter = BirchMurnaghanFitter(
    V0_bounds=(0.7, 1.4),
    B0_bounds=(30, 600),
    B0_prime_bounds=(2.0, 7.0)
)

results = fitter.analyze(
    'data/original_phase.csv',
    'data/new_phase.csv',
    'output/custom_params'
)
```

### 示例 3: 手动输入数据

```python
import numpy as np

fitter = BirchMurnaghanFitter()

# 手动输入数据
V_orig = np.array([16.8, 16.5, 16.2, 15.9, 15.6])
P_orig = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
V_new = np.array([15.5, 15.2, 14.9, 14.6])
P_new = np.array([15.0, 20.0, 25.0, 30.0])

fitter.set_data_manually(V_orig, P_orig, V_new, P_new)

# 执行拟合
fitter.fit_all_phases()

# 绘图
fitter.plot_pv_curves('output/pv_curves.png')
fitter.plot_residuals('output/residuals.png')
fitter.save_results_to_csv('output/results.csv')
```

### 示例 4: 分步操作

```python
fitter = BirchMurnaghanFitter()

# 步骤1: 加载数据
fitter.load_data_from_csv('data/original.csv', 'data/new.csv')

# 步骤2: 拟合
results_orig, results_new = fitter.fit_all_phases()

# 步骤3: 可视化
fitter.plot_pv_curves(save_path='output/pv_curves.png')
fitter.plot_residuals(save_path='output/residuals.png')

# 步骤4: 保存结果
fitter.save_results_to_csv('output/results.csv')
```

### 示例 5: 只拟合单相

```python
import numpy as np

fitter = BirchMurnaghanFitter()

V = np.array([16.8, 16.5, 16.2, 15.9, 15.6])
P = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

results = fitter.fit_single_phase(V, P, "My Phase")

print(f"V₀ = {results['2nd_order']['V0']:.4f} Å³/atom")
print(f"B₀ = {results['2nd_order']['B0']:.2f} GPa")
```

### 示例 6: 批量处理

```python
samples = ['sampleA', 'sampleB', 'sampleC']

for sample in samples:
    fitter = BirchMurnaghanFitter()
    results = fitter.analyze(
        f'data/{sample}_original.csv',
        f'data/{sample}_new.csv',
        f'output/{sample}'
    )
```

### 示例 7: 只显示不保存

```python
fitter = BirchMurnaghanFitter()

# 不指定output_dir，只显示图表
results = fitter.analyze(
    'data/original.csv',
    'data/new.csv',
    output_dir=None
)
```

### 示例 8: 使用静态方法

```python
from birch_murnaghan_fitter import BirchMurnaghanFitter
import numpy as np

# 已知参数
V0 = 16.8
B0 = 150
B0_prime = 4.0

# 计算一系列压力
volumes = np.linspace(14.0, 16.8, 20)

# 2阶BM
pressures_2nd = [BirchMurnaghanFitter.birch_murnaghan_2nd(V, V0, B0)
                 for V in volumes]

# 3阶BM
pressures_3rd = [BirchMurnaghanFitter.birch_murnaghan_3rd(V, V0, B0, B0_prime)
                 for V in volumes]
```

---

## 📊 CSV文件格式要求

输入CSV文件必须包含以下列:

```csv
Pressure (GPa),V_atomic
0.00,16.8000
5.00,16.5000
10.00,16.2000
15.00,15.9000
```

- `Pressure (GPa)`: 压力（单位：GPa）
- `V_atomic`: 原子体积（单位：Å³/atom）

---

## 📈 输出文件

程序会自动生成以下文件（如果指定了output_dir）:

1. **BM_fitting_results.png** - P-V曲线拟合图（4个子图）
2. **BM_fitting_residuals.png** - 残差分析图（4个子图）
3. **BM_fitting_parameters.csv** - 拟合参数汇总表

### 输出CSV格式

```csv
Phase,Fitting Order,V₀ (Å³/atom),V₀ Error,B₀ (GPa),B₀ Error,B₀',B₀' Error,R²,RMSE (GPa)
Original Phase,2nd Order,16.850000,0.010000,145.5000,2.3000,4.000000,0.000000,0.99850000,0.150000
Original Phase,3rd Order,16.840000,0.015000,147.2000,3.1000,4.150000,0.120000,0.99920000,0.120000
New Phase,2nd Order,15.200000,0.012000,165.3000,2.8000,4.000000,0.000000,0.99800000,0.180000
New Phase,3rd Order,15.190000,0.018000,166.8000,3.5000,4.080000,0.150000,0.99880000,0.140000
```

---

## 📊 访问结果

```python
# 执行分析
results = fitter.analyze('data/original.csv', 'data/new.csv', 'output')

# 访问原始相2阶BM结果
orig_2nd = results['original_phase']['2nd_order']
print(f"V₀ = {orig_2nd['V0']:.4f} ± {orig_2nd['V0_err']:.4f} Å³/atom")
print(f"B₀ = {orig_2nd['B0']:.2f} ± {orig_2nd['B0_err']:.2f} GPa")
print(f"R² = {orig_2nd['R_squared']:.6f}")
print(f"RMSE = {orig_2nd['RMSE']:.4f} GPa")

# 访问新相3阶BM结果
new_3rd = results['new_phase']['3rd_order']
print(f"V₀ = {new_3rd['V0']:.4f} Å³/atom")
print(f"B₀ = {new_3rd['B0']:.2f} GPa")
print(f"B₀' = {new_3rd['B0_prime']:.3f}")

# 获取拟合的压力数据
fitted_pressures = orig_2nd['fitted_P']
```

---

## ⚙️ 完整工作流程示例

```python
from birch_murnaghan_fitter import BirchMurnaghanFitter
import numpy as np

# 第1步：创建拟合器（可选：自定义参数）
fitter = BirchMurnaghanFitter(
    V0_bounds=(0.8, 1.3),
    B0_bounds=(50, 500),
    B0_prime_bounds=(2.5, 6.5),
    max_iterations=10000
)

# 第2步：执行完整分析
results = fitter.analyze(
    original_csv='data/original_phase.csv',
    new_csv='data/new_phase.csv',
    output_dir='output/BM_fitting'
)

# 第3步：分析结果
if results:
    print("\n✓ 分析完成！")

    # 原始相结果
    print(f"\n原始相 (2阶BM):")
    print(f"  V₀ = {results['original_phase']['2nd_order']['V0']:.4f} Å³/atom")
    print(f"  B₀ = {results['original_phase']['2nd_order']['B0']:.2f} GPa")
    print(f"  R² = {results['original_phase']['2nd_order']['R_squared']:.6f}")

    # 新相结果
    print(f"\n新相 (3阶BM):")
    print(f"  V₀ = {results['new_phase']['3rd_order']['V0']:.4f} Å³/atom")
    print(f"  B₀ = {results['new_phase']['3rd_order']['B0']:.2f} GPa")
    print(f"  B₀' = {results['new_phase']['3rd_order']['B0_prime']:.3f}")
    print(f"  R² = {results['new_phase']['3rd_order']['R_squared']:.6f}")
```

---

## ❓ 快速参考表

| 任务 | 代码 |
|------|------|
| 创建拟合器 | `fitter = BirchMurnaghanFitter()` |
| 完整分析 | `results = fitter.analyze(orig_csv, new_csv, output_dir)` |
| 加载数据 | `fitter.load_data_from_csv(orig_csv, new_csv)` |
| 手动设置数据 | `fitter.set_data_manually(V_orig, P_orig, V_new, P_new)` |
| 拟合所有相 | `results_o, results_n = fitter.fit_all_phases()` |
| 拟合单相 | `results = fitter.fit_single_phase(V, P, name)` |
| 绘制P-V曲线 | `fitter.plot_pv_curves(save_path)` |
| 绘制残差 | `fitter.plot_residuals(save_path)` |
| 保存结果 | `fitter.save_results_to_csv(output_path)` |
| 2阶BM计算 | `P = BirchMurnaghanFitter.birch_murnaghan_2nd(V, V0, B0)` |
| 3阶BM计算 | `P = BirchMurnaghanFitter.birch_murnaghan_3rd(V, V0, B0, B0')` |

---

## 🔍 常见问题

### Q1: 拟合失败怎么办？

**A**: 尝试以下方法:
1. 检查数据质量（是否有异常值）
2. 调整参数边界（`V0_bounds`, `B0_bounds`, `B0_prime_bounds`）
3. 增加最大迭代次数（`max_iterations`）
4. 确保数据点数足够（至少3-4个点）

### Q2: 如何判断2阶还是3阶拟合更好？

**A**: 查看以下指标:
- **R²值**: 越接近1越好
- **RMSE**: 越小越好
- **物理合理性**: B0'通常在3-6之间
- 如果3阶拟合R²提升不明显，2阶已足够

### Q3: 如何修改参数边界？

**A**: 在初始化时指定:
```python
fitter = BirchMurnaghanFitter(
    V0_bounds=(0.7, 1.4),      # 更宽的V0范围
    B0_bounds=(30, 600),        # 更宽的B0范围
    B0_prime_bounds=(2.0, 7.0)  # 更宽的B0'范围
)
```

### Q4: CSV文件格式不对怎么办？

**A**: 确保CSV包含 `Pressure (GPa)` 和 `V_atomic` 两列。如果列名不同，可以先用pandas重命名:
```python
import pandas as pd
df = pd.read_csv('data.csv')
df = df.rename(columns={'压力': 'Pressure (GPa)', '体积': 'V_atomic'})
df.to_csv('data_renamed.csv', index=False)
```

---

## 🚀 总结

**最简单的三行代码**:
```python
fitter = BirchMurnaghanFitter()
results = fitter.analyze('original.csv', 'new.csv', 'output')
print(f"B₀ = {results['original_phase']['2nd_order']['B0']:.2f} GPa")
```

查看 `bm_example_usage.py` 了解更多示例！
