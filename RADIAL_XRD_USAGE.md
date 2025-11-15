# Radial XRD 方位角积分功能使用指南

## 功能概述

Radial XRD 模块提供了对衍射环进行选择性方位角积分的功能，可以对一系列 H5 数据集进行批量处理。

## 方位角定义

**重要：** 本模块使用以下方位角约定：

```
       90° (↑)
         |
180° (←)--+--→ 0° (Right Horizontal)
         |
      270° (↓)
```

- **0°** = 右侧水平方向（3点钟位置）
- **90°** = 顶部（12点钟位置）
- **180°** = 左侧（9点钟位置）
- **270°** = 底部（6点钟位置）
- **角度方向：** 从右侧水平方向开始，逆时针增加

## 如何在脚本中调用

### 1. 导入模块

```python
from azimuthal_integration import AzimuthalIntegrator, get_preset_sectors
```

### 2. 初始化积分器

```python
# 基本初始化（仅 PONI 文件）
integrator = AzimuthalIntegrator(
    poni_path="/path/to/your/calibration.poni"
)

# 带 mask 文件的初始化
integrator = AzimuthalIntegrator(
    poni_path="/path/to/your/calibration.poni",
    mask_path="/path/to/your/mask.edf"  # 或 .npy 文件
)
```

### 3. 单扇区积分

#### 方法 A: 单个 H5 文件积分

```python
import h5py

# 读取 H5 文件
with h5py.File("your_data.h5", 'r') as f:
    data = f['entry/data/data'][()]

# 对单个扇区进行积分 (例如: 0° 到 90°)
x, intensity = integrator.integrate_azimuthal_range(
    data=data,
    azimuth_start=0,      # 起始角度
    azimuth_end=90,       # 结束角度
    npt=4000,            # 积分点数
    unit='2th_deg'       # 单位: '2th_deg', 'q_A^-1', 'q_nm^-1', 'r_mm'
)

# 保存结果
import pandas as pd
df = pd.DataFrame({'2th_deg': x, 'Intensity': intensity})
df.to_csv('output_0_90.csv', index=False)
```

#### 方法 B: 批量处理多个 H5 文件

```python
# 批量处理一系列 H5 文件
output_files = integrator.batch_integrate_h5(
    input_pattern="/path/to/data/*.h5",
    output_dir="/path/to/output",
    azimuth_start=0,
    azimuth_end=90,
    npt=4000,
    unit='2th_deg',
    dataset_path='entry/data/data',  # H5 文件中的数据路径
    sector_label='Sector_0_90'       # 扇区标签（用于文件命名）
)

print(f"Generated {len(output_files)} files")
```

### 4. 多扇区积分

#### 使用预设配置

```python
# 获取预设扇区配置
sector_list = get_preset_sectors('quadrants')
# 返回: [(0, 90, 'Q1_Right'), (90, 180, 'Q2_Top'),
#        (180, 270, 'Q3_Left'), (270, 360, 'Q4_Bottom')]

# 批量处理多个扇区
output_files = integrator.batch_integrate_multiple_sectors(
    input_pattern="/path/to/data/*.h5",
    output_dir="/path/to/output",
    sector_list=sector_list,
    npt=4000,
    unit='2th_deg',
    dataset_path='entry/data/data'
)
```

#### 自定义扇区配置

```python
# 定义自定义扇区
custom_sectors = [
    (30, 60, 'Custom_30_60'),      # 30° 到 60°
    (120, 150, 'Custom_120_150'),  # 120° 到 150°
    (210, 240, 'Custom_210_240')   # 210° 到 240°
]

# 批量处理
output_files = integrator.batch_integrate_multiple_sectors(
    input_pattern="/path/to/data/*.h5",
    output_dir="/path/to/output",
    sector_list=custom_sectors,
    npt=4000,
    unit='2th_deg'
)
```

## 预设扇区配置

### 1. Quadrants (四象限)
```python
sector_list = get_preset_sectors('quadrants')
# 4 个扇区:
# - Q1_Right: 0° to 90°
# - Q2_Top: 90° to 180°
# - Q3_Left: 180° to 270°
# - Q4_Bottom: 270° to 360°
```

### 2. Octants (八分区)
```python
sector_list = get_preset_sectors('octants')
# 8 个扇区: 每 45° 一个扇区 (0-45°, 45-90°, ...)
```

### 3. Hemispheres (半球)
```python
sector_list = get_preset_sectors('hemispheres')
# 2 个扇区:
# - Hemisphere_Right: 0° to 180°
# - Hemisphere_Left: 180° to 360°
```

### 4. Horizontal & Vertical (水平 & 垂直)
```python
sector_list = get_preset_sectors('horizontal_vertical')
# 4 个扇区:
# - Horizontal_Right: 315° to 45°
# - Horizontal_Left: 135° to 225°
# - Vertical_Top: 45° to 135°
# - Vertical_Bottom: 225° to 315°
```

## 在 GUI 中使用

### 1. 启动 GUI
```python
python curve_fitting.py
```

### 2. 切换到 Radial XRD 标签页
点击顶部的 "Radial" 标签

### 3. 填写参数

#### Integration Settings (积分设置)
- **PONI File**: 选择校准文件
- **Mask File**: （可选）选择掩码文件
- **Input Pattern**: 选择输入 H5 文件（支持通配符）
- **Output Directory**: 选择输出目录
- **Dataset Path**: H5 文件中的数据路径（默认: `entry/data/data`）
- **Number of Points**: 积分点数（默认: 4000）
- **Unit**: 单位选择（2th_deg, q_A^-1, q_nm^-1, r_mm）

#### Azimuthal Angle Settings (方位角设置)

**模式 1: Single Sector (单扇区)**
- **Start Angle (°)**: 起始角度 (0-360)
- **End Angle (°)**: 结束角度 (0-360)
- **Sector Label**: 扇区标签（用于文件命名）

**模式 2: Multiple Sectors (多扇区)**
- 选择预设配置（quadrants, octants, hemispheres, horizontal_vertical）

### 4. 运行
点击 "Run Azimuthal Integration" 按钮

## 特殊情况处理

### 跨越角处理 (Wrap-around)
如果需要积分跨越 0° 的区域（例如：350° 到 10°），程序会自动处理：

```python
# 例子: 积分 350° 到 10° 的区域
x, I = integrator.integrate_azimuthal_range(
    data=data,
    azimuth_start=350,
    azimuth_end=10,
    npt=4000,
    unit='2th_deg'
)
# 程序会自动将其分为两个区域: 350°-360° 和 0°-10°，然后合并结果
```

## 输出文件格式

### 单扇区模式
文件名格式: `{原文件名}_{扇区标签}_azim_{起始角度}_{结束角度}.csv`

例如:
- `data_001_Sector_1_azim_0_90.csv`
- `data_002_Sector_1_azim_0_90.csv`

### 多扇区模式
文件名格式: `{原文件名}_{扇区标签}.csv`

例如 (quadrants 预设):
- `data_001_Q1_Right.csv`
- `data_001_Q2_Top.csv`
- `data_001_Q3_Left.csv`
- `data_001_Q4_Bottom.csv`

### CSV 文件内容
```csv
2th_deg,Intensity
10.5,1234.56
10.51,1235.78
...
```

## 完整示例脚本

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的方位角积分示例
"""

from azimuthal_integration import AzimuthalIntegrator, get_preset_sectors

# 1. 初始化积分器
integrator = AzimuthalIntegrator(
    poni_path="/path/to/calibration.poni",
    mask_path="/path/to/mask.edf"
)

# 2. 示例 1: 单扇区批量处理
print("=" * 60)
print("示例 1: 单扇区 (0° to 90°)")
print("=" * 60)

output_files_1 = integrator.batch_integrate_h5(
    input_pattern="/path/to/data/*.h5",
    output_dir="/path/to/output/single_sector",
    azimuth_start=0,
    azimuth_end=90,
    npt=4000,
    unit='2th_deg',
    sector_label='Right_Quadrant'
)

print(f"\n生成了 {len(output_files_1)} 个文件\n")

# 3. 示例 2: 四象限批量处理
print("=" * 60)
print("示例 2: 四象限")
print("=" * 60)

quadrant_sectors = get_preset_sectors('quadrants')

output_files_2 = integrator.batch_integrate_multiple_sectors(
    input_pattern="/path/to/data/*.h5",
    output_dir="/path/to/output/quadrants",
    sector_list=quadrant_sectors,
    npt=4000,
    unit='2th_deg'
)

print(f"\n生成了 {len(output_files_2)} 个文件\n")

# 4. 示例 3: 自定义扇区
print("=" * 60)
print("示例 3: 自定义扇区")
print("=" * 60)

custom_sectors = [
    (0, 30, 'Sector_A'),
    (90, 120, 'Sector_B'),
    (180, 210, 'Sector_C'),
    (270, 300, 'Sector_D')
]

output_files_3 = integrator.batch_integrate_multiple_sectors(
    input_pattern="/path/to/data/*.h5",
    output_dir="/path/to/output/custom",
    sector_list=custom_sectors,
    npt=4000,
    unit='q_A^-1'  # 使用 q 单位
)

print(f"\n生成了 {len(output_files_3)} 个文件\n")

print("=" * 60)
print("所有处理完成！")
print("=" * 60)
```

## 常见问题

### Q1: 如何确定我需要的方位角范围？
**A:** 根据你的样品和实验设置：
- 如果样品是各向同性的，可以使用任意扇区
- 如果样品有方向性（如纤维、薄膜），选择与样品取向相关的角度
- 使用 Dioptas 或其他软件先可视化数据，确定感兴趣的角度区域

### Q2: 可以同时处理多个不同的角度范围吗？
**A:** 可以！使用多扇区模式或自定义扇区列表：
```python
custom_sectors = [
    (0, 45, 'Sector_1'),
    (90, 135, 'Sector_2'),
    (180, 225, 'Sector_3')
]
```

### Q3: 输出的单位有什么区别？
**A:**
- `2th_deg`: 2θ 角度（度）
- `q_A^-1`: 散射矢量 q，单位 Å⁻¹
- `q_nm^-1`: 散射矢量 q，单位 nm⁻¹
- `r_mm`: 径向距离，单位 mm

### Q4: Mask 文件是必需的吗？
**A:** 不是必需的。如果没有 mask 文件，初始化时传入 `None` 或不提供该参数即可。

### Q5: 如何处理大量数据文件？
**A:** 使用批量处理功能和通配符：
```python
integrator.batch_integrate_h5(
    input_pattern="/data/experiment_*/scan_*.h5",
    output_dir="/output",
    ...
)
```

## 技术细节

### 依赖库
- `pyFAI`: 用于方位角积分
- `h5py`: 读取 HDF5 文件
- `numpy`: 数值计算
- `pandas`: 数据处理和 CSV 输出
- `fabio`: 读取掩码文件（.edf 格式）

### 性能优化建议
1. 使用合适的 `npt` 值（过大会降低速度，过小会损失分辨率）
2. 如果不需要掩码，不要加载 mask 文件
3. 批量处理时使用 SSD 硬盘存储输出文件

## 更新日志

### Version 1.0 (2025-11-15)
- 初始版本
- 支持单扇区和多扇区积分
- 预设配置：quadrants, octants, hemispheres, horizontal_vertical
- GUI 集成
- 批量处理功能
- 跨越角处理

## 联系方式

如有问题或建议，请联系：candicewang928@gmail.com
