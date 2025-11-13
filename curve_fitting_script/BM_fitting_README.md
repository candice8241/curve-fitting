# Birch-Murnaghan方程拟合PV曲线使用说明

## 功能简介

本程序用于对高压实验的压力-体积(PV)数据进行Birch-Murnaghan状态方程拟合,可以：

1. 读取原相和新相的PV数据（CSV格式）
2. 使用二阶和三阶Birch-Murnaghan方程拟合PV曲线
3. 计算体模量相关参数：V₀（零压体积）、B₀（零压体模量）、B₀'（体模量一阶导数）
4. 生成高质量的PV曲线图和残差分析图
5. 输出详细的拟合参数和统计信息

## Birch-Murnaghan方程

### 二阶BM方程
```
P = (3B₀/2) × [(V₀/V)^(7/3) - (V₀/V)^(5/3)]
```
其中B₀' = 4（固定值）

### 三阶BM方程
```
P = (3B₀/2) × [(V₀/V)^(7/3) - (V₀/V)^(5/3)] × [1 + (3/4)(B₀' - 4) × ((V₀/V)^(2/3) - 1)]
```

## 数据要求

### 输入文件格式
- 文件类型：CSV格式
- 必需列：
  - `V_atomic`: 平均原子体积 (Å³/atom)
  - `Pressure (GPa)`: 压力值 (GPa)

### 示例数据文件
```
V_atomic,Pressure (GPa)
16.5432,0.5
16.2341,2.3
15.9876,5.1
15.6543,8.7
...
```

## 使用方法

### 1. 准备数据文件
确保你有以下两个CSV文件：
- `all_results_original_peaks_lattice.csv` - 原相数据
- `all_results_new_peaks_lattice.csv` - 新相数据

### 2. 修改数据路径
打开 `bm_fitting.py`，在 `main()` 函数中修改数据目录：

```python
# 设置数据路径（请根据实际情况修改）
data_dir = r"D:\HEPS\ID31\dioptas_data\Al0"  # 修改为你的数据目录
```

### 3. 运行程序
```bash
python bm_fitting.py
```

### 4. 查看结果
程序会在数据目录下创建 `BM_fitting_output` 文件夹，包含：
- `BM_fitting_results.png` - PV曲线拟合图
- `BM_fitting_residuals.png` - 残差分析图
- `BM_fitting_parameters.csv` - 拟合参数汇总表

## 拟合参数范围

为避免过拟合和获得物理上合理的结果，本程序对拟合参数设置了以下约束：

### V₀（零压体积）
- 范围：最大实验体积的 0.8 - 1.3 倍
- 理由：零压体积应略大于或等于最大实验体积

### B₀（零压体模量）
- 范围：50 - 500 GPa
- 理由：涵盖大多数常见材料的体模量范围
  - 软材料（有机物、分子晶体）：~5-50 GPa
  - 一般无机材料：50-200 GPa
  - 硬材料（氧化物、碳化物）：200-500 GPa
  - 超硬材料（金刚石）：>400 GPa

### B₀'（体模量一阶导数）
- 二阶BM：固定为 4.0（理论值）
- 三阶BM：2.5 - 6.5
- 理由：基于大量实验数据统计
  - 大多数材料：3 - 6
  - 典型值：~4
  - 异常值：金属氢、某些高压相可能>6或<3

## 参考文献

1. Birch, F. (1947). "Finite Elastic Strain of Cubic Crystals". Physical Review. 71 (11): 809-824.

2. Birch, F. (1978). "Finite strain isotherm and velocities for single-crystal and polycrystalline NaCl at high pressures and 300K". Journal of Geophysical Research. 83 (B3): 1257.

3. Angel, R.J. (2000). "Equations of State". Reviews in Mineralogy and Geochemistry. 41 (1): 35-59.

4. Holzapfel, W.B. (1996). "Physics of solids under strong compression". Reports on Progress in Physics. 59 (1): 29-90.

## 输出解释

### 控制台输出
```
原相 - 三阶Birch-Murnaghan拟合结果:
============================================================
V₀ = 16.8432 ± 0.0123 Å³/atom    # 零压体积及其标准误差
B₀ = 156.34 ± 3.21 GPa           # 零压体模量及其标准误差
B₀' = 4.234 ± 0.156              # 体模量一阶导数及其标准误差
R² = 0.998765                    # 决定系数（越接近1拟合越好）
RMSE = 0.2341 GPa                # 均方根误差（越小拟合越好）
```

### 拟合质量评估
- **R² > 0.99**: 优秀拟合
- **R² = 0.95-0.99**: 良好拟合
- **R² < 0.95**: 需要检查数据质量或模型选择

### 选择二阶还是三阶？
1. **数据点较少（<8个）**: 使用二阶BM方程，避免过拟合
2. **压力范围较小（<20 GPa）**: 二阶和三阶差异不大，二阶更稳定
3. **压力范围较大（>20 GPa）或材料可压缩性高**: 三阶BM更准确
4. **判断标准**:
   - 比较RMSE：选择较小的
   - 比较残差分布：选择更均匀、无系统性偏差的
   - 如果三阶的B₀'接近4，说明二阶已足够

## 常见问题

### Q1: 拟合失败怎么办？
- 检查数据中是否有异常值或缺失值
- 确保压力和体积数据配对正确
- 尝试手动调整初始猜测值

### Q2: 拟合参数不合理？
- 检查数据质量（测量误差、数据点分布）
- 考虑压力标定是否准确
- 检查是否存在相变或非静水压影响

### Q3: 二阶和三阶结果差异很大？
- 可能数据点不足以支持三阶拟合（过拟合）
- 可能存在系统性误差
- 建议使用二阶结果或增加数据点

### Q4: 如何判断是否过拟合？
- 三阶拟合的B₀'误差很大（相对误差>20%）
- 残差图显示无规律的随机分布
- 拟合曲线在数据范围外出现非物理行为

## 依赖库

```bash
pip install numpy pandas matplotlib scipy
```

## 作者信息

作者：candicewang928@gmail.com
创建日期：2025-11-13
版本：1.0

## 许可证

Apache License 2.0
