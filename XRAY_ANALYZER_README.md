# XRayDiffractionAnalyzer 类使用说明

## 目录
1. [简介](#简介)
2. [快速开始](#快速开始)
3. [类结构](#类结构)
4. [详细API](#详细api)
5. [使用示例](#使用示例)
6. [参数说明](#参数说明)

---

## 简介

`XRayDiffractionAnalyzer` 是一个用于X射线衍射数据分析的Python类，主要功能包括:

- 📊 **相变识别**: 自动检测压力诱导的相变点
- 🔍 **峰位跟踪**: 跟踪新峰和原始峰在不同压力下的演化
- 📐 **晶格拟合**: 支持8种晶体系统的晶格参数拟合
- 📈 **体积计算**: 自动计算晶胞体积和原子体积
- 💾 **结果导出**: 将分析结果保存为CSV文件

---

## 快速开始

### 最简单的用法（3行代码）

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
results = analyzer.analyze('your_data.csv', original_system='cubic_FCC',
                          new_system='Hexagonal', auto_mode=True)
```

### 交互模式（程序会提示选择晶体系统）

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
results = analyzer.analyze('your_data.csv')  # 程序会提示您选择晶体系统
```

---

## 类结构

### 初始化参数

```python
XRayDiffractionAnalyzer(
    wavelength=0.4133,          # X射线波长 (Å)
    peak_tolerance_1=0.3,       # 相变识别容差 (度)
    peak_tolerance_2=0.4,       # 新峰确定容差 (度)
    peak_tolerance_3=0.01,      # 新峰跟踪容差 (度)
    n_pressure_points=4         # 稳定新峰所需压力点数
)
```

### 支持的晶体系统

| 代码 | 名称 | 最少峰数 | 每晶胞原子数 |
|------|------|----------|--------------|
| `cubic_FCC` | 面心立方 (FCC) | 1 | 4 |
| `cubic_BCC` | 体心立方 (BCC) | 1 | 2 |
| `cubic_SC` | 简单立方 (SC) | 1 | 1 |
| `Hexagonal` | 六方密排 (HCP) | 2 | 2 |
| `Tetragonal` | 四方 | 2 | 1 |
| `Orthorhombic` | 正交 | 3 | 1 |
| `Monoclinic` | 单斜 | 4 | 1 |
| `Triclinic` | 三斜 | 6 | 1 |

---

## 详细API

### 主要方法

#### 1. `analyze()` - 完整分析流程

```python
results = analyzer.analyze(
    csv_path,                    # CSV文件路径
    original_system='cubic_FCC', # 原始相晶体系统（可选）
    new_system='Hexagonal',      # 新相晶体系统（可选）
    auto_mode=True               # 是否自动模式（True=不交互）
)
```

**返回值**:
```python
{
    'original_results': {
        压力1: {'a': ..., 'V_cell': ..., 'V_atomic': ...},
        压力2: {...},
        ...
    },
    'new_results': {
        压力1: {'a': ..., 'c': ..., 'V_cell': ..., 'V_atomic': ...},
        ...
    },
    'transition_pressure': 15.2  # 相变压力 (GPa)
}
```

#### 2. `read_pressure_peak_data()` - 读取数据

```python
pressure_data = analyzer.read_pressure_peak_data('data.csv')
# 返回: {压力1: [峰1, 峰2, ...], 压力2: [...], ...}
```

#### 3. `find_phase_transition_point()` - 识别相变

```python
transition_p, before_p, after_p = analyzer.find_phase_transition_point()
# 返回: (相变压力, 相变前压力列表, 相变后压力列表)
```

#### 4. `fit_lattice_parameters()` - 拟合晶格参数

```python
results = analyzer.fit_lattice_parameters(
    peak_dataset,         # 峰位数据集
    crystal_system_key    # 晶体系统代码
)
```

### 静态方法（工具函数）

这些方法不需要创建实例即可使用:

```python
# 2theta ↔ d spacing 转换
d = XRayDiffractionAnalyzer.two_theta_to_d(30.0, wavelength=0.4133)
two_theta = XRayDiffractionAnalyzer.d_to_two_theta(2.5, wavelength=0.4133)

# 计算d spacing
d = XRayDiffractionAnalyzer.calculate_d_cubic((1,1,1), a=4.05)
d = XRayDiffractionAnalyzer.calculate_d_hexagonal((1,0,1), a=3.0, c=5.0)
d = XRayDiffractionAnalyzer.calculate_d_tetragonal((1,0,1), a=3.0, c=4.0)
d = XRayDiffractionAnalyzer.calculate_d_orthorhombic((1,0,1), a=3.0, b=4.0, c=5.0)

# 计算晶胞体积
V = XRayDiffractionAnalyzer.calculate_cell_volume_cubic(a=4.05)
V = XRayDiffractionAnalyzer.calculate_cell_volume_hexagonal(a=3.0, c=5.0)
V = XRayDiffractionAnalyzer.calculate_cell_volume_tetragonal(a=3.0, c=4.0)
V = XRayDiffractionAnalyzer.calculate_cell_volume_orthorhombic(a=3.0, b=4.0, c=5.0)
```

---

## 使用示例

### 示例 1: 自动模式（推荐）

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

# 创建分析器
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

# 运行分析（FCC → HCP 相变）
results = analyzer.analyze(
    csv_path='data.csv',
    original_system='cubic_FCC',
    new_system='Hexagonal',
    auto_mode=True
)

# 访问结果
print(f"相变压力: {results['transition_pressure']:.2f} GPa")

for pressure, params in results['original_results'].items():
    print(f"{pressure:.2f} GPa: a = {params['a']:.6f} Å")
```

### 示例 2: 交互模式

```python
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

# 程序会提示您选择晶体系统
results = analyzer.analyze('data.csv')
```

### 示例 3: 分步操作（高级）

```python
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

# 第1步: 读取数据
pressure_data = analyzer.read_pressure_peak_data('data.csv')

# 第2步: 识别相变
transition_p, before_p, after_p = analyzer.find_phase_transition_point()

# 第3步: 获取新峰
transition_peaks = pressure_data[transition_p]
prev_peaks = pressure_data[before_p[-1]]

tolerance_windows = [(p - analyzer.peak_tolerance_1,
                     p + analyzer.peak_tolerance_1) for p in prev_peaks]

new_peaks = [peak for peak in transition_peaks
             if not any(lower <= peak <= upper
                       for (lower, upper) in tolerance_windows)]

# 第4步: 跟踪新峰
stable_count, tracked_new_peaks = analyzer.collect_tracked_new_peaks(
    pressure_data, transition_p, after_p, new_peaks, analyzer.peak_tolerance_2
)

# 第5步: 构建原始峰数据集
original_peak_dataset = analyzer.build_original_peak_dataset(
    pressure_data, tracked_new_peaks, analyzer.peak_tolerance_3
)

# 第6步: 拟合晶格参数
original_results = analyzer.fit_lattice_parameters(
    original_peak_dataset, 'cubic_FCC'
)

new_results = analyzer.fit_lattice_parameters(
    tracked_new_peaks, 'Hexagonal'
)

# 第7步: 保存结果
analyzer.save_lattice_results_to_csv(
    original_results, 'original_phase.csv', 'cubic_FCC'
)
analyzer.save_lattice_results_to_csv(
    new_results, 'new_phase.csv', 'Hexagonal'
)
```

### 示例 4: 批量处理

```python
csv_files = ['sample1.csv', 'sample2.csv', 'sample3.csv']
systems = {
    'sample1.csv': ('cubic_FCC', 'Hexagonal'),
    'sample2.csv': ('cubic_BCC', 'cubic_FCC'),
    'sample3.csv': ('Hexagonal', 'cubic_FCC')
}

all_results = {}
for csv_file in csv_files:
    analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
    orig_sys, new_sys = systems[csv_file]

    results = analyzer.analyze(
        csv_file,
        original_system=orig_sys,
        new_system=new_sys,
        auto_mode=True
    )
    all_results[csv_file] = results
```

### 示例 5: 单相分析（无相变）

```python
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

# 读取数据
pressure_data = analyzer.read_pressure_peak_data('single_phase.csv')

# 直接拟合（假设为FCC）
results = analyzer.fit_lattice_parameters(pressure_data, 'cubic_FCC')

# 保存结果
analyzer.save_lattice_results_to_csv(
    results, 'single_phase_results.csv', 'cubic_FCC'
)
```

### 示例 6: 访问中间结果

```python
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

results = analyzer.analyze(
    'data.csv',
    original_system='cubic_FCC',
    new_system='Hexagonal',
    auto_mode=True
)

# 访问存储的数据
print("所有压力点:", list(analyzer.pressure_data.keys()))
print("相变压力:", analyzer.transition_pressure)
print("相变前压力:", analyzer.before_pressures)
print("相变后压力:", analyzer.after_pressures)

# 访问原始峰数据
for pressure, data in analyzer.original_peak_dataset.items():
    print(f"{pressure:.2f} GPa: {data['count']} 个原始峰")

# 访问新峰数据
for pressure, peaks in analyzer.tracked_new_peaks.items():
    print(f"{pressure:.2f} GPa: {len(peaks)} 个新峰")
```

---

## 参数说明

### CSV 文件格式要求

输入CSV文件必须包含以下列:

- `File`: 压力值或包含压力信息的文件名（如 "15.2" 或 "sample_15.2GPa"）
- `Center`: 峰位置（2theta角度，单位：度）

示例:
```csv
File,Center
0.5,12.345
0.5,25.678
0.5,38.901

10.2,12.456
10.2,25.789
10.2,38.012
```

### 容差参数说明

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `peak_tolerance_1` | 0.3° | 用于识别相变点：如果新峰与旧峰的2θ差异超过此值，则认为发生了相变 | 0.2-0.5° |
| `peak_tolerance_2` | 0.4° | 用于确定新峰数量：在后续压力点中追踪新峰时的匹配范围 | 0.3-0.6° |
| `peak_tolerance_3` | 0.01° | 用于精确追踪：分离新峰和原始峰时的精确匹配范围 | 0.01-0.05° |

**调整建议**:
- 如果峰较宽或数据噪声大，增大容差值
- 如果峰较窄且数据质量好，减小容差值
- 如果错过相变点，尝试增大 `peak_tolerance_1`
- 如果误判相变点，尝试减小 `peak_tolerance_1`

### 输出结果说明

#### 立方晶系输出

```python
{
    压力: {
        'a': 晶格常数 (Å),
        'V_cell': 晶胞体积 (Å³),
        'V_atomic': 原子体积 (Å³/atom),
        'num_peaks_used': 使用的峰数
    }
}
```

#### 六方晶系输出

```python
{
    压力: {
        'a': 晶格常数a (Å),
        'c': 晶格常数c (Å),
        'c/a': c/a比值,
        'V_cell': 晶胞体积 (Å³),
        'V_atomic': 原子体积 (Å³/atom),
        'num_peaks_used': 使用的峰数
    }
}
```

#### 四方晶系输出

```python
{
    压力: {
        'a': 晶格常数a (Å),
        'c': 晶格常数c (Å),
        'c/a': c/a比值,
        'V_cell': 晶胞体积 (Å³),
        'V_atomic': 原子体积 (Å³/atom),
        'num_peaks_used': 使用的峰数
    }
}
```

#### 正交晶系输出

```python
{
    压力: {
        'a': 晶格常数a (Å),
        'b': 晶格常数b (Å),
        'c': 晶格常数c (Å),
        'V_cell': 晶胞体积 (Å³),
        'V_atomic': 原子体积 (Å³/atom),
        'num_peaks_used': 使用的峰数
    }
}
```

---

## 常见问题

### Q1: 如何选择合适的晶体系统？

**A**: 根据材料的晶体结构选择:
- 大多数金属: FCC, BCC, 或 HCP
- 参考文献中的晶体结构
- 使用ICDD/PDF卡片数据库

### Q2: 程序未检测到相变怎么办？

**A**: 尝试以下方法:
1. 增大 `peak_tolerance_1` 参数
2. 检查CSV文件格式是否正确
3. 确认数据质量（是否有足够的压力点）
4. 手动检查峰位数据

### Q3: 拟合结果不合理怎么办？

**A**: 检查:
1. 晶体系统选择是否正确
2. hkl指标顺序是否匹配
3. 峰位数据质量
4. 是否有足够的峰用于拟合

### Q4: 如何修改每晶胞原子数？

**A**: 在代码中修改 `CRYSTAL_SYSTEMS` 字典:
```python
XRayDiffractionAnalyzer.CRYSTAL_SYSTEMS['cubic_FCC']['atoms_per_cell'] = 4
```

### Q5: 可以分析多个相变吗？

**A**: 当前版本仅支持单个相变点识别。如需分析多个相变，需要:
1. 将数据分段
2. 对每段独立运行分析

---

## 完整调用格式总结

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

# ============ 方式1: 最简单（自动模式）============
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
results = analyzer.analyze('data.csv', original_system='cubic_FCC',
                          new_system='Hexagonal', auto_mode=True)

# ============ 方式2: 交互模式 ============
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
results = analyzer.analyze('data.csv')  # 会提示选择晶体系统

# ============ 方式3: 自定义参数 ============
analyzer = XRayDiffractionAnalyzer(
    wavelength=0.5000,
    peak_tolerance_1=0.5,
    peak_tolerance_2=0.6,
    peak_tolerance_3=0.02,
    n_pressure_points=3
)
results = analyzer.analyze('data.csv', original_system='cubic_BCC',
                          new_system='Hexagonal', auto_mode=True)

# ============ 方式4: 分步操作（高级）============
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
pressure_data = analyzer.read_pressure_peak_data('data.csv')
transition_p, before_p, after_p = analyzer.find_phase_transition_point()
# ... 其他步骤见"示例3"

# ============ 方式5: 使用静态方法 ============
d = XRayDiffractionAnalyzer.two_theta_to_d(30.0, wavelength=0.4133)
V = XRayDiffractionAnalyzer.calculate_cell_volume_cubic(a=4.05)
```

---

## 技术支持

如有问题或建议，请联系作者或提交Issue。

---

**版本**: 1.0
**最后更新**: 2025-11-13
