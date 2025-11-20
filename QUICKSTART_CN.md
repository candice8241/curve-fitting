# 🚀 快速开始 - XRD应用打包

**5分钟内将您的Python GUI打包成exe文件！**

---

## ⚡ 三步快速打包

### 第1步：准备项目文件

确保您的项目目录包含以下文件：

```
✅ main.py              - 主程序（已创建）
✅ xrd_app.spec         - 打包配置（已创建）
✅ build.bat            - 打包脚本（已创建）
⚠️  theme_module.py     - 需要您提供
⚠️  powder_module.py    - 需要您提供
⚠️  radial_module.py    - 需要您提供
⚠️  single_crystal_module.py - 需要您提供
📁 resources/           - 资源文件夹（已创建）
   └── app_icon.ico     - 应用图标（可选）
```

**重要**：如果缺少模块文件，请将它们复制到项目根目录。

### 第2步：安装依赖

打开命令提示符（CMD）或PowerShell，运行：

```bash
pip install -r requirements_gui.txt
```

或者手动安装：

```bash
pip install pyinstaller numpy scipy matplotlib pandas pillow
```

### 第3步：执行打包

双击运行 `build.bat`，等待完成！

```
📂 输出位置：dist/XRD_PostProcessing/XRD_PostProcessing.exe
```

---

## 🖱️ 创建桌面快捷方式

### 方法A：直接拖拽

1. 打开 `dist/XRD_PostProcessing/` 文件夹
2. 找到 `XRD_PostProcessing.exe`
3. 按住 **Alt** 键，用鼠标拖动到桌面
4. 松手，快捷方式创建完成！

### 方法B：右键菜单

1. 右键点击 `XRD_PostProcessing.exe`
2. 选择 **发送到** → **桌面快捷方式**
3. 完成！

---

## 📋 检查清单

在打包前，请确认：

- [ ] Python 3.8+ 已安装
- [ ] PyInstaller 已安装 (`pip install pyinstaller`)
- [ ] 所有模块文件存在（theme_module.py等）
- [ ] 图标文件已放入 resources/ 文件夹（可选）
- [ ] 所有依赖已安装

---

## ⚠️ 如果缺少模块文件

如果您还没有创建 `theme_module.py` 等文件，有两个选择：

### 选项1：创建空模块（临时测试）

创建基本的模块文件用于测试打包：

**theme_module.py:**
```python
import tkinter as tk

class GUIBase:
    def __init__(self):
        self.colors = {
            'bg': '#F5F5F5',
            'card_bg': '#FFFFFF',
            'text_dark': '#333333'
        }

class ModernButton(tk.Button):
    def __init__(self, parent, text, command):
        super().__init__(parent, text=text, command=command)

class ModernTab(tk.Frame):
    def __init__(self, parent, text, command, is_active=False):
        super().__init__(parent)
        self.is_active = is_active

    def set_active(self, active):
        self.is_active = active

class CuteSheepProgressBar(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
```

**powder_module.py, radial_module.py, single_crystal_module.py:**
```python
import tkinter as tk

class PowderXRDModule:  # 或其他模块名
    def __init__(self, parent, root):
        self.parent = parent
        self.root = root

    def setup_ui(self):
        tk.Label(self.parent, text="模块功能开发中...").pack(pady=20)
```

### 选项2：从完整项目复制

如果您在其他位置有完整的模块文件，请将它们复制到项目根目录。

---

## 🎯 一键命令（高级用户）

如果您熟悉命令行，可以一次性完成所有操作：

```bash
# 安装依赖并打包
pip install -r requirements_gui.txt && pyinstaller --clean xrd_app.spec
```

---

## 🔧 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| 找不到Python | 安装Python 3.8+ (https://www.python.org/) |
| 找不到模块 | 运行 `pip install -r requirements_gui.txt` |
| 打包失败 | 检查是否缺少模块文件 |
| 无法运行exe | 右键 → 以管理员身份运行 |
| 图标不显示 | 确保 resources/app_icon.ico 存在且格式正确 |

---

## 📞 需要帮助？

查看完整文档：`BUILD_GUIDE.md`

联系我们：
- lixd@ihep.ac.cn
- fzhang@ihep.ac.cn
- yswang@ihep.ac.cn

---

**开始打包吧！💜✨ 只需要3分钟！**
