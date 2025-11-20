# 🎨 XRD峰拟合工具 - 图标自定义

## 📌 快速开始

### 最简单的方法：使用自动生成的图标

```bash
# 1. 安装Pillow（如果还没安装）
pip install Pillow

# 2. 生成图标
python generate_icons.py

# 3. 运行程序（图标会自动加载）
python peak_fitting_gui_enhanced.py
```

就这么简单！✨

## 🎯 支持的图标格式

- ✅ `.ico` - Windows图标（推荐用于Windows）
- ✅ `.png` - PNG格式（跨平台）
- ✅ `.gif` - GIF格式

## 📁 文件说明

### 核心文件
- **`icon_utils.py`** - 图标工具库，提供图标加载和生成功能
- **`generate_icons.py`** - 一键生成所有尺寸的XRD图标
- **`create_simple_icon.py`** - 简单图标创建器（不依赖Pillow）

### 文档
- **`ICON_CUSTOMIZATION_GUIDE.md`** - 完整的图标自定义指南
- **`ICON_README.md`** - 本文件，快速参考

### 生成的图标文件（运行generate_icons.py后）
- `xrd_icon.ico` - Windows图标
- `xrd_icon.png` - 默认图标（64x64）
- `xrd_icon_16.png` 至 `xrd_icon_256.png` - 各种尺寸

## 🚀 使用方法

### 方法1: 自动使用（推荐）

将图标文件放在脚本同目录下，程序会自动检测并加载：

```
curve-fitting/
├── peak_fitting_gui_enhanced.py
├── xrd_icon.ico
└── xrd_icon.png
```

### 方法2: 指定图标路径

```python
from peak_fitting_gui_enhanced import main

# 使用自定义图标
main(icon_path='path/to/your/icon.ico')
```

### 方法3: 在代码中设置

```python
import tkinter as tk
from peak_fitting_gui_enhanced import PeakFittingGUI

root = tk.Tk()
app = PeakFittingGUI(root, icon_path='my_icon.png')
root.mainloop()
```

## 🎨 图标设计特点

自动生成的图标包含：
- 🟣 紫色渐变背景（匹配应用主题）
- ⭕ 衍射环图案（代表X射线衍射）
- ➕ 金色十字准星（光束中心标记）
- 📝 "XRD"文字标识

## 💡 常见问题

### Q: 图标不显示怎么办？

A: 按以下步骤检查：
1. 确认图标文件存在：`ls -l xrd_icon.*`
2. 确认文件名正确：`xrd_icon.ico` 或 `xrd_icon.png`
3. 查看程序输出，会显示图标加载状态
4. 尝试使用完整路径指定图标

### Q: 如何使用自己的图标？

A: 两种方式：
1. 将你的图标重命名为 `xrd_icon.png` 或 `xrd_icon.ico`
2. 在代码中指定路径：`main(icon_path='your_icon.png')`

### Q: 需要安装什么依赖？

A: 仅用于生成图标时需要Pillow：
```bash
pip install Pillow
```

程序运行本身不需要Pillow（会使用内嵌的备用图标）。

### Q: 支持哪些尺寸？

A: 推荐尺寸：
- 小图标：16x16, 32x32
- 标准图标：48x48, 64x64
- 高清图标：128x128, 256x256

程序会自动缩放到合适的尺寸。

## 🛠️ 开发者信息

### 图标搜索顺序

程序按以下顺序搜索图标：

1. 自定义指定路径
2. 当前工作目录：`xrd_icon.ico`, `xrd_icon.png`
3. 脚本所在目录：`<script_dir>/xrd_icon.ico`
4. icons子目录：`<script_dir>/icons/xrd_icon.ico`
5. 内嵌默认图标（始终可用）

### API参考

#### set_window_icon()
```python
def set_window_icon(window, icon_path=None, use_default=True):
    """
    设置窗口图标

    Parameters:
    -----------
    window : tk.Tk or tk.Toplevel
        要设置图标的窗口
    icon_path : str, optional
        自定义图标路径
    use_default : bool
        是否使用默认图标（当找不到自定义图标时）

    Returns:
    --------
    bool : 是否成功设置图标
    """
```

#### create_simple_xrd_icon()
```python
def create_simple_xrd_icon(output_path='xrd_icon.png', size=64):
    """
    创建简单的XRD主题图标

    Parameters:
    -----------
    output_path : str
        输出路径
    size : int
        图标尺寸（正方形）

    Returns:
    --------
    bool : 是否成功创建
    """
```

## 📝 快速命令参考

```bash
# 生成所有尺寸图标
python generate_icons.py

# 生成特定尺寸
python icon_utils.py --size 128 --output my_icon.png

# 生成所有变体
python icon_utils.py --all

# 创建简单图标（无Pillow）
python create_simple_icon.py

# 检查图标文件
ls -l xrd_icon.*

# 安装依赖
pip install Pillow
```

## 🔗 相关链接

- [完整图标指南](ICON_CUSTOMIZATION_GUIDE.md)
- [集成指南](INTEGRATION_GUIDE.md)
- [快速开始](QUICK_START.md)

## ✅ 测试检查清单

使用图标前请确认：

- [ ] 图标文件已生成或准备好
- [ ] 文件命名正确（`xrd_icon.ico` 或 `xrd_icon.png`）
- [ ] 文件放在正确位置（脚本同目录）
- [ ] 图标尺寸适当（建议64x64或更大）
- [ ] 运行程序，确认图标正常显示
- [ ] 检查窗口标题栏图标
- [ ] 检查任务栏图标

## 🎉 完成！

现在你的XRD峰拟合工具拥有了专业的自定义图标！

如果遇到任何问题，请查看 [ICON_CUSTOMIZATION_GUIDE.md](ICON_CUSTOMIZATION_GUIDE.md) 获取详细帮助。

---

**Made with** 💜 **by XRD Analysis Team**
