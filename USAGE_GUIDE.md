# XRayDiffractionAnalyzer 调用格式速查

## 📋 基本调用格式

### ⭐ 最简单的调用方式（推荐）

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

# 创建分析器
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

# 一行代码完成分析
results = analyzer.analyze(
    csv_path='your_data.csv',
    original_system='cubic_FCC',    # 原始相：面心立方
    new_system='Hexagonal',         # 新相：六方密排
    auto_mode=True                  # 自动模式，无需交互
)
```

---

## 📦 初始化参数

```python
analyzer = XRayDiffractionAnalyzer(
    wavelength=0.4133,          # X射线波长 (Å)
    peak_tolerance_1=0.3,       # 相变识别容差 (度)
    peak_tolerance_2=0.4,       # 新峰确定容差 (度)
    peak_tolerance_3=0.01,      # 新峰跟踪容差 (度)
    n_pressure_points=4         # 稳定新峰所需压力点数
)
```

---

## 🎯 晶体系统代码

| 代码 | 晶体系统 | 代码 | 晶体系统 |
|------|----------|------|----------|
| `'cubic_FCC'` | 面心立方 | `'Tetragonal'` | 四方 |
| `'cubic_BCC'` | 体心立方 | `'Orthorhombic'` | 正交 |
| `'cubic_SC'` | 简单立方 | `'Monoclinic'` | 单斜 |
| `'Hexagonal'` | 六方密排 | `'Triclinic'` | 三斜 |

---

## 💡 常用调用示例

### 1️⃣ FCC → HCP 相变

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
results = analyzer.analyze('data.csv', original_system='cubic_FCC',
                          new_system='Hexagonal', auto_mode=True)
```

### 2️⃣ BCC → FCC 相变

```python
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
results = analyzer.analyze('data.csv', original_system='cubic_BCC',
                          new_system='cubic_FCC', auto_mode=True)
```

### 3️⃣ 交互模式（程序会提示选择）

```python
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
results = analyzer.analyze('data.csv')  # 不指定晶体系统，程序会提示
```

### 4️⃣ 单相分析（无相变）

```python
analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
pressure_data = analyzer.read_pressure_peak_data('data.csv')
results = analyzer.fit_lattice_parameters(pressure_data, 'cubic_FCC')
analyzer.save_lattice_results_to_csv(results, 'output.csv', 'cubic_FCC')
```

### 5️⃣ 批量处理多个文件

```python
files = ['sample1.csv', 'sample2.csv', 'sample3.csv']

for csv_file in files:
    analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)
    results = analyzer.analyze(csv_file, original_system='cubic_FCC',
                              new_system='Hexagonal', auto_mode=True)
```

---

## 📊 访问结果

```python
# 运行分析
results = analyzer.analyze('data.csv', original_system='cubic_FCC',
                          new_system='Hexagonal', auto_mode=True)

# 访问相变压力
print(f"相变压力: {results['transition_pressure']:.2f} GPa")

# 访问原始相结果
for pressure, params in results['original_results'].items():
    print(f"压力 {pressure:.2f} GPa:")
    print(f"  晶格常数 a = {params['a']:.6f} Å")
    print(f"  晶胞体积 V = {params['V_cell']:.6f} Å³")
    print(f"  原子体积 = {params['V_atomic']:.6f} Å³/atom")

# 访问新相结果
for pressure, params in results['new_results'].items():
    print(f"压力 {pressure:.2f} GPa:")
    print(f"  晶格常数 a = {params['a']:.6f} Å")
    print(f"  晶格常数 c = {params['c']:.6f} Å")
    print(f"  c/a 比值 = {params['c/a']:.6f}")
    print(f"  晶胞体积 V = {params['V_cell']:.6f} Å³")
    print(f"  原子体积 = {params['V_atomic']:.6f} Å³/atom")
```

---

## 🔧 静态方法（工具函数）

不需要创建实例即可使用:

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

# 角度与d spacing转换
d = XRayDiffractionAnalyzer.two_theta_to_d(30.0, wavelength=0.4133)
two_theta = XRayDiffractionAnalyzer.d_to_two_theta(2.5, wavelength=0.4133)

# 计算d spacing
d_cubic = XRayDiffractionAnalyzer.calculate_d_cubic((1,1,1), a=4.05)
d_hex = XRayDiffractionAnalyzer.calculate_d_hexagonal((1,0,1), a=3.0, c=5.0)

# 计算晶胞体积
V_cubic = XRayDiffractionAnalyzer.calculate_cell_volume_cubic(a=4.05)
V_hex = XRayDiffractionAnalyzer.calculate_cell_volume_hexagonal(a=3.0, c=5.0)
```

---

## 📝 CSV文件格式

输入文件必须包含 `File` 和 `Center` 两列:

```csv
File,Center
0.5,12.345
0.5,25.678
0.5,38.901

10.2,12.456
10.2,25.789
```

- `File`: 压力值（GPa）或包含压力的文件名
- `Center`: 峰位置（2theta，单位：度）
- 空行分隔不同压力点（可选）

---

## ⚙️ 分步调用（高级用法）

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)

# 第1步: 读取数据
pressure_data = analyzer.read_pressure_peak_data('data.csv')

# 第2步: 识别相变
transition_p, before_p, after_p = analyzer.find_phase_transition_point()

# 第3步: 识别新峰
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

---

## 🎨 自定义参数示例

### 调整容差参数

```python
# 适用于峰较宽或噪声大的数据
analyzer = XRayDiffractionAnalyzer(
    wavelength=0.4133,
    peak_tolerance_1=0.5,    # 增大相变识别容差
    peak_tolerance_2=0.6,    # 增大新峰确定容差
    peak_tolerance_3=0.02,   # 增大新峰跟踪容差
    n_pressure_points=3      # 减少所需压力点数
)
results = analyzer.analyze('data.csv', original_system='cubic_FCC',
                          new_system='Hexagonal', auto_mode=True)
```

### 不同波长

```python
# Synchrotron光源，λ = 0.5000 Å
analyzer = XRayDiffractionAnalyzer(wavelength=0.5000)

# Cu Kα，λ = 1.5406 Å
analyzer = XRayDiffractionAnalyzer(wavelength=1.5406)

# Mo Kα，λ = 0.7107 Å
analyzer = XRayDiffractionAnalyzer(wavelength=0.7107)
```

---

## 📂 输出文件

程序会自动生成以下文件:

1. `原始文件名_original_peaks_lattice.csv` - 原始相晶格参数
2. `原始文件名_new_peaks_lattice.csv` - 新相晶格参数

### 输出CSV格式示例

**立方晶系:**
```csv
Pressure (GPa),a,V_cell,V_atomic,num_peaks_used
0.50,4.050000,66.430125,16.607531,5
10.20,3.980000,63.044792,15.761198,5
```

**六方晶系:**
```csv
Pressure (GPa),a,c,c/a,V_cell,V_atomic,num_peaks_used
15.50,2.950000,4.800000,1.627119,36.119382,18.059691,6
20.30,2.920000,4.750000,1.626712,35.095823,17.547911,6
```

---

## 🚀 完整工作流程示例

```python
from xray_diffraction_analyzer import XRayDiffractionAnalyzer

# 创建分析器
analyzer = XRayDiffractionAnalyzer(
    wavelength=0.4133,          # 设置波长
    peak_tolerance_1=0.3,       # 相变识别容差
    peak_tolerance_2=0.4,       # 新峰确定容差
    peak_tolerance_3=0.01,      # 新峰跟踪容差
    n_pressure_points=4         # 稳定新峰所需压力点数
)

# 执行分析
results = analyzer.analyze(
    csv_path='my_xrd_data.csv',      # 输入文件
    original_system='cubic_FCC',      # 原始相晶体系统
    new_system='Hexagonal',           # 新相晶体系统
    auto_mode=True                    # 自动模式
)

# 输出结果
if results and 'transition_pressure' in results:
    print(f"\n✓ 分析完成！")
    print(f"相变压力: {results['transition_pressure']:.2f} GPa")
    print(f"原始相数据点: {len(results['original_results'])}")
    print(f"新相数据点: {len(results['new_results'])}")
    print(f"\n结果已保存到CSV文件。")
else:
    print("\n✓ 单相分析完成！")
```

---

## ❓ 快速参考

| 任务 | 代码 |
|------|------|
| 创建分析器 | `analyzer = XRayDiffractionAnalyzer(wavelength=0.4133)` |
| 完整分析 | `results = analyzer.analyze('data.csv', original_system='cubic_FCC', new_system='Hexagonal', auto_mode=True)` |
| 读取数据 | `pressure_data = analyzer.read_pressure_peak_data('data.csv')` |
| 识别相变 | `transition_p, before_p, after_p = analyzer.find_phase_transition_point()` |
| 拟合晶格 | `results = analyzer.fit_lattice_parameters(peak_dataset, 'cubic_FCC')` |
| 保存结果 | `analyzer.save_lattice_results_to_csv(results, 'output.csv', 'cubic_FCC')` |
| 角度转d | `d = XRayDiffractionAnalyzer.two_theta_to_d(30.0, 0.4133)` |
| d转角度 | `angle = XRayDiffractionAnalyzer.d_to_two_theta(2.5, 0.4133)` |

---

**快速上手**: 复制第一段代码，修改文件路径和晶体系统即可使用！
