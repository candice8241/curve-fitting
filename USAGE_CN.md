# 相变分析程序使用说明

## 概述

这是一个用于晶体相变分析的Python程序，可以：
1. 从XRD峰位数据中识别相变点
2. 根据峰位判断晶系类型
3. 计算晶胞参数
4. 使用Birch-Murnaghan状态方程拟合P-V曲线

## 安装依赖

```bash
pip install -r requirements.txt
```

## 输入文件格式

程序接受CSV格式的输入文件，包含两列：
- `file`: 压力值（单位：GPa）
- `center`: 峰位（2theta角度）

不同压力点之间用空行分隔。

### 示例格式：

```csv
file,center
5.2,38.45
5.2,44.71
5.2,65.08

10.5,38.52
10.5,44.82
10.5,65.22

15.8,38.61
15.8,44.95
15.8,65.39
```

## 使用方法

### 1. 基本使用

```python
from phase_transition_analysis import analyze_phase_transition

# 分析相变
analyze_phase_transition(
    csv_file="your_data.csv",  # 输入CSV文件路径
    output_dir="./output"       # 输出目录
)
```

### 2. 参数配置

在程序开头可以调整以下参数：

```python
PEAK_TOLERANCE_1 = 0.3   # 初始新峰检测的容差（2theta度）
PEAK_TOLERANCE_2 = 0.2   # 稳定新峰计数的容差
PEAK_TOLERANCE_3 = 0.15  # 相变后追踪新峰的容差
NUM_STABLE_POINTS = 4    # 确认稳定新峰所需的压力点数量
WAVELENGTH = 0.6199      # X射线波长（埃）
```

### 3. 修改波长

如果您使用不同的X射线源，请修改 `WAVELENGTH` 参数：

```python
# Cu Kα
WAVELENGTH = 1.5406

# Mo Kα
WAVELENGTH = 0.7107

# 同步辐射（示例）
WAVELENGTH = 0.6199
```

## 程序工作原理

### 1. 相变识别

程序按以下步骤识别相变点：

1. 将所有压力点按从小到大排序
2. 比较相邻压力点的峰位
3. 当出现超过 `PEAK_TOLERANCE_1` 的新峰时，标记为潜在相变点
4. 向后检查 `NUM_STABLE_POINTS` 个压力点
5. 如果新峰数量稳定（最后3个点相同），确认相变

### 2. 晶系判断

支持的晶系及所需最少峰数：

| 晶系 | 所需最少峰数 | 晶胞参数 |
|------|------------|---------|
| Cubic (立方) | 1 | a |
| Hexagonal (六方) | 2 | a, c |
| Tetragonal (四方) | 2 | a, c |
| Orthorhombic (正交) | 3 | a, b, c |
| Rhombohedral (菱方) | 2 | a, α |
| Monoclinic (单斜) | 4 | a, b, c, β |
| Triclinic (三斜) | 6 | a, b, c, α, β, γ |

### 3. 晶胞参数计算

程序使用不同晶系的d-spacing公式：

- **立方系**: d = a / √(h² + k² + l²)
- **六方系**: 1/d² = 4/3 × (h² + hk + k²)/a² + l²/c²
- **四方系**: 1/d² = (h² + k²)/a² + l²/c²
- **正交系**: 1/d² = h²/a² + k²/b² + l²/c²

### 4. EOS拟合

使用Birch-Murnaghan状态方程：

**二阶BM方程**:
```
P = (3/2) × K₀ × [(V₀/V)^(7/3) - (V₀/V)^(5/3)]
```

**三阶BM方程**:
```
P = (3/2) × K₀ × [(V₀/V)^(7/3) - (V₀/V)^(5/3)] × {1 + (3/4) × (K₀' - 4) × [(V₀/V)^(2/3) - 1]}
```

其中：
- V₀: 零压体积
- K₀: 体积模量
- K₀': 体积模量的压力导数

## 输出结果

程序会生成以下文件：

### 1. 原相P-V曲线图
`original_phase_PV.png` - 显示原始相的压力-体积关系及拟合曲线

### 2. 新相P-V曲线图
`new_phase_PV.png` - 显示新相的压力-体积关系及拟合曲线

### 3. 组合P-V曲线图
`combined_PV.png` - 显示两相的P-V曲线对比

### 4. 终端输出

程序会在终端打印详细的分析结果：

```
==============================================================
PHASE TRANSITION ANALYSIS
==============================================================

✓ Loaded 10 pressure points
  Pressure range: 5.20 - 42.90 GPa

✓ Phase transition detected at P = 21.30 GPa
  New peak evolution: [1, 2, 2, 2]

==============================================================
ORIGINAL PHASE ANALYSIS
==============================================================
✓ Crystal system: CUBIC
  Lattice parameters: {'a': 4.0521}

  Fitting EOS for original phase (4 points)...

  2nd Order BM EOS:
    V0 = 16.5432 Å³/atom
    K0 = 165.23 GPa

  3rd Order BM EOS:
    V0 = 16.5589 Å³/atom
    K0 = 163.87 GPa
    K0' = 4.152

==============================================================
NEW PHASE ANALYSIS
==============================================================
  Number of new peaks: 2
  New peak positions: [42.18 51.24]

✓ New phase crystal system: HEXAGONAL
  Lattice parameters: {'a': 2.6532, 'c': 4.3215, 'c/a': 1.6289}

  Fitting EOS for new phase (5 points)...

  2nd Order BM EOS:
    V0 = 10.2341 Å³/atom
    K0 = 185.67 GPa

  3rd Order BM EOS:
    V0 = 10.2456 Å³/atom
    K0 = 184.23 GPa
    K0' = 3.987
```

## 调整晶系判断

### 修改每个晶系的原子数

如果您知道具体的结构类型，可以修改：

```python
'cubic': {
    'atoms_per_cell': {
        'fcc': 4,   # 面心立方
        'bcc': 2,   # 体心立方
        'sc': 1     # 简单立方
    }
}
```

### 修改hkl序列

每个晶系的hkl序列已按2theta从小到大预设。如需修改：

```python
'cubic': {
    'hkl_sequence': [
        (1,1,1), (2,0,0), (2,2,0), (3,1,1), (2,2,2),
        # 添加更多hkl...
    ]
}
```

## 高级用法

### 单独使用各个功能模块

```python
from phase_transition_analysis import (
    identify_crystal_system,
    calculate_unit_cell_volume,
    fit_eos_3rd_order
)

# 识别晶系
import numpy as np
peaks = np.array([38.45, 44.71, 65.08])
phase = identify_crystal_system(peaks)
print(f"Crystal system: {phase.crystal_system}")

# 计算体积
volume = calculate_unit_cell_volume(
    phase.lattice_params,
    phase.crystal_system
)

# 拟合EOS
pressures = np.array([5, 10, 15, 20])
volumes = np.array([16.5, 15.8, 15.2, 14.7])
eos_params = fit_eos_3rd_order(pressures, volumes)
print(f"K0 = {eos_params['K0']:.2f} GPa")
```

## 常见问题

### 1. 没有检测到相变

可能原因：
- `PEAK_TOLERANCE_1` 设置过大
- 数据点太少
- 相变不明显

解决方法：减小容差值或增加数据点

### 2. 晶系识别错误

可能原因：
- 峰位精度不够
- hkl序列不正确
- 波长设置错误

解决方法：检查波长设置和峰位数据质量

### 3. EOS拟合失败

可能原因：
- 数据点太少（至少需要3个点）
- 数据分散度太大

解决方法：增加压力点或检查体积计算

## 参考文献

1. Birch, F. (1947). Finite Elastic Strain of Cubic Crystals. Physical Review, 71(11), 809-824.
2. Angel, R. J. (2000). Equations of State. Reviews in Mineralogy and Geochemistry, 41(1), 35-59.
3. Toby, B. H., & Von Dreele, R. B. (2013). GSAS-II: the genesis of a modern open-source all purpose crystallography software package. Journal of Applied Crystallography, 46(2), 544-549.

## 联系方式

如有问题或建议，请联系：candicewang928@gmail.com
