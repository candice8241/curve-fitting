# XRD数据处理GUI程序使用说明

## 概述
这是一个完整的XRD（X射线衍射）数据处理套件，包含多个功能模块用于峰拟合、相变分析、体积计算和Birch-Murnaghan状态方程拟合。

## 安装依赖
```bash
pip install numpy pandas scipy matplotlib tkinter
```

## 启动程序
```bash
python xrd_gui.py
```

## CSV文件格式要求

### 峰位数据CSV格式
程序现在支持以下CSV格式：

**第一行（列标题）：**
```
pressure (gpa), peak_1_2theta, peak_2_theta, peak_3_2theta, ..., number of peaks
```

**数据行示例：**
```
5.2, 12.34, 15.67, 18.92, 0, 0, 3
7.8, 12.45, 15.78, 18.03, 21.45, 0, 4
10.1, 12.56, 15.89, 19.14, 21.56, 23.78, 5
```

**格式说明：**
1. **pressure (gpa)** 或 **pressure(gpa)** - 压力列（单位：GPa）
   - 列名可以是 "pressure (gpa)", "Pressure", "pressure(gpa)" 等，程序会自动识别

2. **peak_1_2theta, peak_2_2theta, peak_3_2theta, ...** - 峰位列
   - 每列代表一个峰的2theta角度值
   - 如果某压力点下峰少于最大峰数，用0或留空表示
   - 程序会自动过滤掉值为0或空的峰

3. **number of peaks** - 峰的数量（可选）
   - 这一列主要用于参考，程序会自动统计实际峰数

### 完整示例CSV文件

```csv
pressure (gpa),peak_1_2theta,peak_2_2theta,peak_3_2theta,peak_4_2theta,peak_5_2theta,number of peaks
0.0,12.34,15.67,18.92,0,0,3
2.5,12.38,15.71,18.96,0,0,3
5.0,12.42,15.75,19.00,21.34,0,4
7.5,12.46,15.79,19.04,21.38,23.56,5
10.0,12.50,15.83,19.08,21.42,23.60,5
```

## 主要功能模块

### 1. 峰分离（Separate Original & New Peaks）
- **输入**：包含峰位数据的CSV文件
- **功能**：自动识别相变压力点，分离原始峰和新出现的峰
- **输出**：
  - `*_original_peaks_dataset.csv` - 原始相的峰位数据
  - `*_new_peaks_dataset.csv` - 新相的峰位数据

### 2. 体积计算（Calculate Volume & Fit Lattice Parameters）
- **输入**：峰位CSV文件
- **参数设置**：
  - Crystal System：晶系选择（FCC, BCC, SC等）
  - Wavelength：X射线波长（Å）
  - Tolerances：峰匹配容差
- **输出**：晶格参数和体积数据

### 3. Birch-Murnaghan拟合
- **输入**：包含Pressure和Volume列的CSV文件
- **参数**：BM Order（2阶或3阶）
- **输出**：
  - V₀（零压体积）
  - K₀（体模量）
  - K₀'（体模量导数）
  - 拟合图表

## 参数说明

### 容差参数（Tolerances）
- **Tol-1**：用于初始相变检测的峰位容差（默认：0.3）
- **Tol-2**：用于追踪新峰的容差（默认：0.4）
- **Tol-3**：用于匹配原始峰的容差（默认：0.01）

### N压力点
- 用于确认相变的连续压力点数量（默认：4）
- 较大的值可以减少误判，但可能遗漏短暂的相变

## 使用流程示例

### 典型工作流程
1. **准备数据**：将峰拟合结果整理成CSV格式
2. **峰分离**：
   - 选择CSV文件
   - 设置容差参数
   - 点击"Separate Original & New Peaks"
3. **体积计算**：
   - 选择分离后的CSV文件
   - 选择晶系
   - 设置波长
   - 点击"Calculate Volume"
4. **BM拟合**：
   - 选择包含P-V数据的CSV
   - 选择BM阶数
   - 点击"Birch-Murnaghan Fit"

## 故障排除

### 常见错误1：CSV列名不匹配
**错误信息**：`CSV file must contain pressure column`

**解决方案**：
- 确保CSV第一行包含列名
- 压力列名应包含"pressure"或"gpa"字样
- 峰位列名应包含"peak"和"2theta"字样

### 常见错误2：峰值格式错误
**解决方案**：
- 确保峰位值都是数字
- 没有峰的位置用0或留空
- 检查数值中是否有非数字字符

### 常见错误3：未检测到相变
**解决方案**：
- 调整容差参数（增大Tol-1）
- 减小N压力点数量
- 检查数据是否确实存在相变

## 文件结构
```
curve-fitting/
├── xrd_gui.py                    # 主GUI程序
├── batch_cal_volume.py           # 体积计算和相变分析模块
├── batch_integration.py          # 1D积分模块
├── peak_fitting.py               # 峰拟合模块
├── birch_murnaghan_batch.py      # BM状态方程拟合模块
├── batch_appearance.py           # GUI外观组件
└── XRD_GUI_README.md             # 本说明文件
```

## 联系与支持
如遇问题，请检查：
1. CSV文件格式是否正确
2. 所有必需的Python包是否已安装
3. 参数设置是否合理

## 更新日志
- 2025-11-14：修改CSV读取功能，支持pressure和peak_X_2theta列格式
- 修复了原有的"File"和"Center"列依赖问题
- 增强了列名识别的灵活性
