# XRD方位角积分工具使用说明

## 功能简介

`xrd_azimuthal_integration.py` 是一个用于XRD衍射环数据方位角积分的Python脚本。它可以：

- ✅ 读取HDF5格式的2D衍射环数据
- ✅ 使用PONI校准文件进行几何校正
- ✅ 支持掩膜（mask）文件来排除坏点或特定区域
- ✅ 批量处理多个文件
- ✅ 输出多种格式的1D积分曲线

## 安装依赖

首先安装所需的Python库：

```bash
pip install -r requirements.txt
```

或单独安装：

```bash
pip install pyFAI h5py numpy fabio scipy matplotlib pandas
```

## 使用方法

### 基本用法

处理单个HDF5文件：

```bash
python xrd_azimuthal_integration.py input.h5 -p calibration.poni -o output_dir/
```

### 带掩膜文件

```bash
python xrd_azimuthal_integration.py input.h5 -p calibration.poni -m mask.npy -o output_dir/
```

### 批量处理

处理多个文件（使用通配符）：

```bash
python xrd_azimuthal_integration.py data/*.h5 -p calibration.poni -o results/
```

或显式列出多个文件：

```bash
python xrd_azimuthal_integration.py file1.h5 file2.h5 file3.h5 -p cal.poni -o output/
```

### 高级参数

指定积分点数、单位和输出格式：

```bash
python xrd_azimuthal_integration.py input.h5 \
  -p calibration.poni \
  -m mask.npy \
  -o output/ \
  --npt 4096 \
  --unit 2th_deg \
  --format dat
```

## 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `input_files` | 输入的HDF5文件路径（支持通配符，如 `*.h5`） |
| `-p, --poni` | PONI校准文件路径 |
| `-o, --output` | 输出目录路径 |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-m, --mask` | None | 掩膜文件路径（支持.npy/.edf/.tif/.h5格式） |
| `--npt` | 2048 | 积分曲线的数据点数 |
| `--unit` | `q_A^-1` | 径向坐标单位，可选：`q_A^-1`, `q_nm^-1`, `2th_deg`, `2th_rad`, `r_mm` |
| `--format` | `xy` | 输出格式，可选：`xy`, `chi`, `dat` |

## 输入文件格式

### HDF5文件

HDF5文件应包含2D衍射图像数据。脚本会自动搜索以下常见数据集路径：

- `entry/data/data`
- `entry/instrument/detector/data`
- `data`
- `image`
- `diffraction`

如果找不到，会使用第一个找到的2D数据集。

### PONI文件

PONI文件是pyFAI的校准文件，包含探测器几何参数。可以使用以下工具生成：

- **pyFAI-calib2**: pyFAI的校准工具
- **Dioptas**: 图形化XRD数据处理软件

示例PONI文件内容：

```
# Detector: Pilatus 2M
Distance: 0.3
Poni1: 0.15
Poni2: 0.16
Rot1: 0.0
Rot2: 0.0
Rot3: 0.0
Wavelength: 1.54e-10
```

### 掩膜文件

掩膜文件用于排除坏像素或感兴趣区域外的像素。支持的格式：

- **.npy**: NumPy数组文件（推荐）
- **.edf**: ESRF数据格式
- **.tif/.tiff**: TIFF图像
- **.h5/.hdf5**: HDF5文件

掩膜应为与衍射图像相同尺寸的2D数组，其中：
- `0` 或 `False` = 有效像素
- `1` 或 `True` = 被掩盖的像素

## 输出格式

### XY格式 (--format xy)

两列文本文件：

```
# Column 1: q (A^-1)
# Column 2: Intensity
0.000000e+00 1.234567e+03
1.000000e-02 2.345678e+03
...
```

### CHI格式 (--format chi)

GSAS-II兼容格式：

```
2-Theta Angle (Degrees)
Intensity
10.0000 1234.5678
10.0500 2345.6789
...
```

### DAT格式 (--format dat)

三列格式（包含误差）：

```
# Column 1: q (A^-1)
# Column 2: Intensity
# Column 3: Error
0.000000e+00 1.234567e+03 3.514598e+01
1.000000e-02 2.345678e+03 4.843221e+01
...
```

## 使用示例

### 示例1：处理单个文件

```bash
python xrd_azimuthal_integration.py \
  sample001.h5 \
  -p LaB6_calibration.poni \
  -o integrated_data/
```

输出：`integrated_data/sample001_integrated.xy`

### 示例2：批量处理含掩膜

```bash
python xrd_azimuthal_integration.py \
  raw_data/*.h5 \
  -p detector_cal.poni \
  -m bad_pixels_mask.npy \
  -o processed/
```

### 示例3：使用2θ角度输出

```bash
python xrd_azimuthal_integration.py \
  diffraction.h5 \
  -p calibration.poni \
  -o results/ \
  --unit 2th_deg \
  --format chi \
  --npt 4096
```

### 示例4：Python脚本调用

也可以在Python脚本中直接使用：

```python
from xrd_azimuthal_integration import XRDAzimuthalIntegrator

# 初始化积分器
integrator = XRDAzimuthalIntegrator(
    poni_file='calibration.poni',
    mask_file='mask.npy'
)

# 处理单个文件
integrator.integrate_file(
    h5_file='sample.h5',
    output_dir='output/',
    npt=2048,
    unit='q_A^-1',
    output_format='xy'
)

# 批量处理
h5_files = ['file1.h5', 'file2.h5', 'file3.h5']
integrator.batch_process(
    h5_files,
    output_dir='output/',
    npt=2048,
    unit='q_A^-1',
    output_format='xy'
)
```

## 常见问题

### Q: 如何获取PONI校准文件？

A: 使用pyFAI-calib2或Dioptas软件对标准样品（如LaB6, CeO2）的衍射图进行校准即可生成。

### Q: 支持哪些HDF5数据结构？

A: 脚本会自动搜索常见的数据集路径。如果你的数据结构不同，可能需要修改 `_read_h5_data` 方法。

### Q: 如何创建掩膜文件？

A: 可以使用以下方法：
- 使用Dioptas创建并导出
- 使用NumPy手动创建并保存为.npy
- 使用pyFAI的掩膜工具

示例Python代码：

```python
import numpy as np

# 创建与探测器尺寸相同的掩膜
mask = np.zeros((1679, 1475), dtype=np.int8)

# 标记坏像素
mask[100:110, 200:210] = 1  # 掩盖某个区域

# 保存
np.save('my_mask.npy', mask)
```

### Q: 积分点数（npt）如何选择？

A:
- 一般使用 2048 或 4096 即可
- 更高的值会有更好的分辨率但文件更大
- 不应超过探测器的径向像素数

### Q: 不同单位的区别？

A:
- `q_A^-1`: 散射矢量 (Å⁻¹)，常用于小角散射
- `q_nm^-1`: 散射矢量 (nm⁻¹)
- `2th_deg`: 衍射角 2θ (度)，最常用
- `2th_rad`: 衍射角 2θ (弧度)
- `r_mm`: 探测器径向距离 (mm)

## 技术细节

- **积分方法**: Split-pixel算法，精确考虑像素分割
- **误差模型**: Poisson统计误差模型
- **处理流程**: 2D衍射图 → 几何校正 → 方位角积分 → 1D曲线

## 相关工具

- **pyFAI**: https://pyfai.readthedocs.io/
- **Dioptas**: https://github.com/Dioptas/Dioptas
- **GSAS-II**: https://subversion.xray.aps.anl.gov/trac/pyGSAS

## 作者

candicewang928@gmail.com

## 许可证

Apache License 2.0
