# 🎨 XRD峰拟合工具 - 图标自定义指南

## 📋 概述

本指南将帮助你自定义XRD峰拟合工具的窗口图标和任务栏图标。

## ⚡ 快速开始

### 方法一: 使用自动生成的图标（推荐）

1. **生成默认XRD图标**
   ```bash
   python icon_utils.py --all
   ```

   这将生成:
   - `xrd_icon.png` (默认图标, 64x64)
   - `xrd_icon.ico` (Windows图标)
   - `xrd_icon_16.png` 到 `xrd_icon_256.png` (多种尺寸)

2. **将图标放在正确位置**
   - 将生成的图标文件放在与Python脚本相同的目录下
   - 程序会自动检测并使用这些图标

3. **运行程序**
   ```bash
   python peak_fitting_gui_enhanced.py
   ```

### 方法二: 使用自己的图标

1. **准备图标文件**
   - 支持格式: `.ico`, `.png`, `.gif`
   - 推荐尺寸: 64x64 或更大
   - 文件名: `xrd_icon.ico` 或 `xrd_icon.png`

2. **放置图标**
   将图标文件放在以下任一位置（按优先级排序）:
   ```
   1. 当前工作目录/xrd_icon.ico
   2. 当前工作目录/xrd_icon.png
   3. 脚本所在目录/xrd_icon.ico
   4. 脚本所在目录/xrd_icon.png
   5. 脚本所在目录/icons/xrd_icon.ico
   6. 脚本所在目录/icons/xrd_icon.png
   ```

3. **程序会自动加载**
   程序启动时会自动搜索并加载图标

## 🎯 详细指南

### 1. 生成自定义图标

#### 使用内置图标生成器

```bash
# 生成所有尺寸的图标
python icon_utils.py --all

# 生成特定尺寸的图标
python icon_utils.py --size 128 --output my_icon.png

# 仅生成默认图标
python icon_utils.py
```

#### 生成的图标特点
- 紫色渐变背景（符合XRD工具主题）
- 衍射环图案（代表X射线衍射）
- 金色十字准星（表示光束中心）
- "XRD"文字标识

### 2. 使用Photoshop/GIMP等工具创建图标

#### 推荐规格:
- **尺寸**: 64x64, 128x128, 或 256x256 像素
- **格式**: PNG（透明背景）或 ICO
- **颜色**: 建议使用紫色系（#BA55D3, #9370DB）保持主题一致

#### 设计建议:
- 使用简洁的图案，避免过于复杂的细节
- 确保在小尺寸下仍然清晰可辨
- 使用对比鲜明的颜色
- 可以加入XRD相关元素：
  - 衍射图案
  - 晶体结构
  - 波形图
  - 峰形曲线

### 3. 转换图标格式

#### PNG 转 ICO (Windows)

使用Python Pillow库:
```python
from PIL import Image

# 打开PNG图片
img = Image.open('my_icon.png')

# 保存为ICO，支持多种尺寸
img.save('my_icon.ico', format='ICO', sizes=[(16,16), (32,32), (48,48), (64,64)])
```

或使用在线工具:
- https://convertio.co/zh/png-ico/
- https://www.icoconverter.com/

## 💻 代码中使用图标

### 在主程序中使用

```python
# 方法1: 使用默认图标（自动搜索）
from peak_fitting_gui_enhanced import main
main()

# 方法2: 指定图标路径
from peak_fitting_gui_enhanced import main
main(icon_path='path/to/your/icon.ico')

# 方法3: 在代码中设置
import tkinter as tk
from peak_fitting_gui_enhanced import PeakFittingGUI

root = tk.Tk()
app = PeakFittingGUI(root, icon_path='my_custom_icon.png')
root.mainloop()
```

### 在集成模块中使用

图标会自动应用到:
1. 主窗口
2. 交互式拟合窗口（Toplevel窗口）
3. 任务栏图标

## 🔍 图标搜索顺序

程序按以下顺序搜索图标:

1. **自定义路径** (如果在代码中指定)
2. **当前目录**:
   - `xrd_icon.ico`
   - `xrd_icon.png`
   - `icon.ico`
   - `icon.png`
3. **脚本目录**:
   - `<script_dir>/xrd_icon.ico`
   - `<script_dir>/xrd_icon.png`
4. **icons子目录**:
   - `<script_dir>/icons/xrd_icon.ico`
   - `<script_dir>/icons/xrd_icon.png`
5. **内嵌默认图标** (如果找不到任何文件)

## 🎨 示例图标模板

### Python代码创建简单图标

```python
from PIL import Image, ImageDraw

# 创建64x64的图标
size = 64
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# 紫色圆形背景
draw.ellipse([0, 0, size, size], fill=(186, 85, 211, 255))

# 白色边框
draw.ellipse([4, 4, size-4, size-4], outline=(255, 255, 255, 255), width=3)

# 中心点
center = size // 2
draw.ellipse([center-4, center-4, center+4, center+4],
            fill=(255, 215, 0, 255))

# 保存
img.save('simple_xrd_icon.png')
img.save('simple_xrd_icon.ico', format='ICO', sizes=[(size, size)])
```

## 🛠️ 故障排查

### 问题1: 图标不显示

**解决方案**:
1. 检查图标文件是否存在:
   ```python
   import os
   print(os.path.exists('xrd_icon.ico'))
   ```

2. 检查文件格式:
   - Windows推荐使用 `.ico` 格式
   - 跨平台使用 `.png` 格式

3. 检查文件权限:
   ```bash
   ls -l xrd_icon.ico
   ```

### 问题2: ICO文件无法加载

**解决方案**:
使用PNG格式代替:
```python
# 将ICO转换为PNG
from PIL import Image
img = Image.open('icon.ico')
img.save('icon.png')
```

### 问题3: 任务栏图标与窗口图标不同

**原因**: Windows系统缓存问题

**解决方案**:
1. 关闭所有程序窗口
2. 清除Windows图标缓存:
   ```cmd
   ie4uinit.exe -show
   ```
3. 重启Windows资源管理器

### 问题4: Pillow未安装

**错误信息**: `ModuleNotFoundError: No module named 'PIL'`

**解决方案**:
```bash
pip install Pillow
```

## 📁 推荐文件结构

```
curve-fitting/
├── peak_fitting_gui_enhanced.py
├── powder_xrd_module_with_interactive_fitting.py
├── icon_utils.py
├── xrd_icon.ico          # Windows图标
├── xrd_icon.png          # 通用图标
└── icons/                # 可选的图标文件夹
    ├── xrd_icon_16.png
    ├── xrd_icon_32.png
    ├── xrd_icon_48.png
    ├── xrd_icon_64.png
    ├── xrd_icon_128.png
    └── xrd_icon_256.png
```

## 🎯 最佳实践

### 1. 图标设计原则
- ✅ 简洁明了，避免细节过多
- ✅ 颜色对比鲜明
- ✅ 与应用主题一致（紫色系）
- ✅ 在小尺寸下仍然清晰

### 2. 文件管理
- ✅ 使用描述性文件名（如 `xrd_icon.ico`）
- ✅ 保留多种尺寸备用
- ✅ 同时提供 `.ico` 和 `.png` 格式
- ✅ 将图标放在 `icons/` 子目录统一管理

### 3. 跨平台兼容性
- Windows: 优先使用 `.ico`
- Mac/Linux: 使用 `.png`
- 提供多种格式确保兼容性

## 📚 参考资源

### 在线图标工具
- **Favicon Generator**: https://realfavicongenerator.net/
- **ICO Convert**: https://www.icoconverter.com/
- **Online Icon Maker**: https://www.iconj.com/ico_icons_maker.php

### 图标设计灵感
- **Icons8**: https://icons8.com/
- **Flaticon**: https://www.flaticon.com/
- **Iconfinder**: https://www.iconfinder.com/

### Python库
- **Pillow**: 图像处理 - `pip install Pillow`
- **cairosvg**: SVG转换 - `pip install cairosvg`

## 🎉 完成检查清单

- [ ] 已生成或准备好图标文件
- [ ] 图标文件已放在正确位置
- [ ] 图标格式正确（.ico 或 .png）
- [ ] 图标尺寸合适（推荐64x64或更大）
- [ ] 程序能正常加载图标
- [ ] 窗口图标显示正确
- [ ] 任务栏图标显示正确
- [ ] 图标设计符合应用主题

## 💡 快速命令参考

```bash
# 生成所有图标
python icon_utils.py --all

# 生成自定义尺寸
python icon_utils.py --size 128 --output my_icon.png

# 运行程序（使用默认图标）
python peak_fitting_gui_enhanced.py

# 运行集成模块测试
python test_integrated_module.py

# 安装Pillow
pip install Pillow

# 检查图标文件
ls -l xrd_icon.*
```

## 🔗 相关文档

- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - 集成指南
- [QUICK_START.md](QUICK_START.md) - 快速开始
- [icon_utils.py](icon_utils.py) - 图标工具源代码

---

**提示**: 如果遇到任何问题，可以检查控制台输出，程序会打印图标加载状态信息。
