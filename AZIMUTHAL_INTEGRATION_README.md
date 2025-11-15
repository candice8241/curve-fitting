# Radial XRD 方位角积分 (Azimuthal Integration)

用于径向X射线衍射数据的方位角积分分析工具。

## 功能特性

- ✅ 单扇区或多扇区方位角积分
- ✅ 预设模板（四象限、八分区、半球等）
- ✅ 自定义角度范围
- ✅ 支持掩膜文件 (.edf, .npy)
- ✅ 批量处理 HDF5 文件
- ✅ 输出 .xy 和 CSV 格式
- ✅ 命令行和 Python API 两种使用方式

## 安装依赖

```bash
pip install pyFAI h5py pandas numpy
```

可选依赖（用于 .edf 掩膜文件）:
```bash
pip install fabio
```

## 方位角坐标系

```
         90° (↑)
          |
180° (←)--+--→ 0°
          |
        270° (↓)

从右侧水平方向（0°）逆时针旋转
```

## 快速开始

### 方法 1: Python API

```python
from radial_xrd_azimuthal_integration import azimuthal_integration, get_preset_sectors

# 单扇区积分
results = azimuthal_integration(
    poni_file="calibration.poni",
    input_pattern="data/*.h5",
    output_dir="results/",
    sectors=[(0, 90, "Q1")],
    npt=4000,
    unit='2th_deg',
    save_csv=True
)

# 使用四象限预设
sectors = get_preset_sectors('quadrants')
results = azimuthal_integration(
    poni_file="calibration.poni",
    input_pattern="data/*.h5",
    output_dir="results/",
    sectors=sectors
)
```

### 方法 2: 命令行

```bash
# 单个扇区
python radial_xrd_azimuthal_integration.py \
    --poni calibration.poni \
    --input "data/*.h5" \
    --output results/ \
    --sector 0 90 "Q1"

# 使用预设模板
python radial_xrd_azimuthal_integration.py \
    --poni calibration.poni \
    --input "data/*.h5" \
    --output results/ \
    --preset quadrants

# 自定义多扇区
python radial_xrd_azimuthal_integration.py \
    --poni calibration.poni \
    --input "data/*.h5" \
    --output results/ \
    --sector 0 90 "Q1" \
    --sector 90 180 "Q2" \
    --sector 180 270 "Q3" \
    --sector 270 360 "Q4"
```

## 详细参数说明

### Python API 参数

#### `azimuthal_integration()` 函数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `poni_file` | str | ✅ | - | PONI 标定文件路径 |
| `input_pattern` | str | ✅ | - | 输入 H5 文件通配符 (如 `"data/*.h5"`) |
| `output_dir` | str | ✅ | - | 输出目录路径 |
| `sectors` | list | ✅ | - | 扇区列表 `[(start, end, label), ...]` |
| `dataset_path` | str | ❌ | `"entry/data/data"` | HDF5 数据集路径 |
| `mask_file` | str | ❌ | `None` | 掩膜文件路径 (.edf 或 .npy) |
| `npt` | int | ❌ | `4000` | 积分点数 |
| `unit` | str | ❌ | `'2th_deg'` | 单位: `'2th_deg'`, `'q_A^-1'`, `'q_nm^-1'`, `'r_mm'` |
| `save_csv` | bool | ❌ | `True` | 是否保存 CSV 文件 |
| `verbose` | bool | ❌ | `True` | 是否显示进度信息 |

**返回值**: `dict` - 字典，键为扇区标签，值为输出文件路径列表

### 命令行参数

```bash
python radial_xrd_azimuthal_integration.py -h
```

**必需参数:**
- `--poni`: PONI 标定文件路径
- `--input`: 输入 H5 文件通配符
- `--output`: 输出目录

**扇区定义 (二选一):**
- `--sector START END LABEL`: 定义扇区（可重复多次）
- `--preset {quadrants|octants|hemispheres|horizontal_vertical}`: 使用预设模板

**可选参数:**
- `--mask`: 掩膜文件路径
- `--dataset`: HDF5 数据集路径（默认: `entry/data/data`）
- `--npt`: 积分点数（默认: 4000）
- `--unit`: 单位（默认: `2th_deg`）
- `--no-csv`: 禁用 CSV 输出
- `--quiet`: 静默模式

## 预设模板

### 1. Quadrants (四象限)

```python
sectors = get_preset_sectors('quadrants')
# [(0, 90, "Q1_0-90"),
#  (90, 180, "Q2_90-180"),
#  (180, 270, "Q3_180-270"),
#  (270, 360, "Q4_270-360")]
```

```
    Q2 | Q1
   ----+----
    Q3 | Q4
```

### 2. Octants (八分区)

```python
sectors = get_preset_sectors('octants')
```

每 45° 一个扇区，共 8 个扇区。

### 3. Hemispheres (半球)

```python
sectors = get_preset_sectors('hemispheres')
# [(0, 180, "Right_Hemisphere"),
#  (180, 360, "Left_Hemisphere")]
```

### 4. Horizontal/Vertical (水平/垂直)

```python
sectors = get_preset_sectors('horizontal_vertical')
# [(0, 90, "Right"),
#  (90, 180, "Top"),
#  (180, 270, "Left"),
#  (270, 360, "Bottom")]
```

## 输出文件

### 1. XY 文件

每个输入文件和每个扇区生成一个 `.xy` 文件:

```
filename_SectorLabel.xy
```

格式:
```
# 2th_deg  Intensity
10.5  1234.5
10.6  1245.2
...
```

### 2. CSV 文件

每个扇区生成一个合并的 CSV 文件:

```
azimuthal_integration_SectorLabel.csv
```

格式:
```
2th_deg,sample1,sample2,sample3,...
10.5,1234.5,1245.2,1256.3,...
10.6,1245.2,1256.3,1267.4,...
...
```

## 使用示例

### 示例 1: 材料各向异性分析

分析轧制材料在不同方向的织构：

```python
from radial_xrd_azimuthal_integration import azimuthal_integration

# 定义轧制方向相关的扇区
sectors = [
    (0, 30, "Rolling_Direction"),       # 轧制方向
    (60, 120, "Transverse_Direction"),  # 横向
    (150, 210, "Opposite_Rolling"),     # 反向轧制
    (240, 300, "Opposite_Transverse")   # 反向横向
]

results = azimuthal_integration(
    poni_file="calibration.poni",
    input_pattern="samples/*.h5",
    output_dir="texture_analysis/",
    sectors=sectors,
    mask_file="mask.edf",
    npt=5000,
    unit='2th_deg',
    save_csv=True
)
```

### 示例 2: 应力应变分析

四象限分析用于应力应变研究：

```python
from radial_xrd_azimuthal_integration import azimuthal_integration, get_preset_sectors

results = azimuthal_integration(
    poni_file="stress_calibration.poni",
    input_pattern="stress_test/*.h5",
    output_dir="stress_results/",
    sectors=get_preset_sectors('quadrants'),
    unit='q_A^-1',
    npt=6000
)
```

### 示例 3: 批量处理多个样品

```python
import os
from radial_xrd_azimuthal_integration import azimuthal_integration, get_preset_sectors

# 样品列表
samples = ['sample1', 'sample2', 'sample3']

for sample in samples:
    print(f"Processing {sample}...")

    results = azimuthal_integration(
        poni_file=f"{sample}/calibration.poni",
        input_pattern=f"{sample}/data/*.h5",
        output_dir=f"results/{sample}/",
        sectors=get_preset_sectors('octants'),
        mask_file=f"{sample}/mask.npy"
    )
```

### 示例 4: 命令行批处理

```bash
#!/bin/bash
# batch_process.sh

PONI="calibration.poni"
MASK="mask.edf"

for sample in sample1 sample2 sample3; do
    echo "Processing ${sample}..."

    python radial_xrd_azimuthal_integration.py \
        --poni "${PONI}" \
        --input "data/${sample}/*.h5" \
        --output "results/${sample}/" \
        --mask "${MASK}" \
        --preset quadrants \
        --npt 5000 \
        --unit 2th_deg
done
```

## 高级用法

### 自定义 HDF5 数据集路径

如果您的 HDF5 文件使用非标准数据集路径：

```python
results = azimuthal_integration(
    poni_file="calibration.poni",
    input_pattern="data/*.h5",
    output_dir="results/",
    sectors=[(0, 90, "Q1")],
    dataset_path="custom/path/to/data"  # 自定义路径
)
```

### 不同的单位系统

```python
# 使用 q 空间 (Å^-1)
results = azimuthal_integration(
    poni_file="calibration.poni",
    input_pattern="data/*.h5",
    output_dir="results_q_space/",
    sectors=get_preset_sectors('quadrants'),
    unit='q_A^-1'
)

# 使用 q 空间 (nm^-1)
results = azimuthal_integration(
    poni_file="calibration.poni",
    input_pattern="data/*.h5",
    output_dir="results_q_nm/",
    sectors=get_preset_sectors('quadrants'),
    unit='q_nm^-1'
)
```

### 精细角度划分

```python
# 每10度一个扇区
sectors = [(i, i+10, f"Sector_{i}-{i+10}") for i in range(0, 360, 10)]

results = azimuthal_integration(
    poni_file="calibration.poni",
    input_pattern="data/*.h5",
    output_dir="fine_division/",
    sectors=sectors,
    npt=8000  # 更高的积分点数
)
```

## 故障排除

### 常见问题

**1. ImportError: No module named 'pyFAI'**
```bash
pip install pyFAI
```

**2. FileNotFoundError: Dataset not found**

检查 HDF5 文件结构并指定正确的 `dataset_path`:
```python
import h5py
with h5py.File('your_file.h5', 'r') as f:
    def print_structure(name):
        print(name)
    f.visit(print_structure)
```

**3. No files found matching pattern**

确保使用引号包围通配符：
```bash
--input "data/*.h5"  # 正确
--input data/*.h5    # 可能错误
```

**4. Mask file format error**

确保掩膜文件是 `.edf` 或 `.npy` 格式。转换示例：
```python
import numpy as np
# 假设 mask 是 numpy 数组
np.save('mask.npy', mask)
```

## 性能优化

### 大批量文件处理

对于大量文件，考虑：

1. **减少积分点数**（如果可接受）:
   ```python
   npt=2000  # 而不是 4000
   ```

2. **禁用详细输出**:
   ```python
   verbose=False
   ```

3. **使用并行处理**（自定义脚本）:
   ```python
   from multiprocessing import Pool

   def process_sector(sector):
       return azimuthal_integration(
           poni_file="cal.poni",
           input_pattern="data/*.h5",
           output_dir=f"results/{sector[2]}/",
           sectors=[sector],
           verbose=False
       )

   with Pool(4) as pool:
       results = pool.map(process_sector, get_preset_sectors('octants'))
   ```

## 引用

如果您在研究中使用此工具，请引用 pyFAI:

> Ashiotis, G., Deschildre, A., Nawaz, Z., Wright, J. P., Karkoulis, D., Picca, F. E., & Kieffer, J. (2015).
> The fast azimuthal integration Python library: pyFAI. Journal of applied crystallography, 48(2), 510-519.

## 许可证

Apache 2.0 License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**更多示例**: 参见 `azimuthal_integration_example.py`
