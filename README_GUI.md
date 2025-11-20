# 🔬 XRD Data Post-Processing - GUI Application

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)]()

**将Python GUI应用轻松打包成桌面应用程序！**

---

## 📦 项目概述

这是一个用于XRD（X射线衍射）数据后处理的图形界面应用程序，支持打包成Windows可执行文件(.exe)和其他平台的独立应用。

### 主要特性

✨ **现代化GUI** - 使用Tkinter构建的美观界面
🔬 **多模块支持** - 粉末XRD、单晶XRD、径向积分
📦 **一键打包** - 轻松打包成exe可执行文件
🎨 **自定义图标** - 支持自定义应用图标
🖥️ **跨平台** - 支持Windows、Linux、macOS

---

## 🚀 快速开始

### 方法1：使用打包脚本（推荐）

```bash
# 1. 安装依赖
pip install -r requirements_gui.txt

# 2. 复制您的模块文件到项目根目录
# 或使用示例模块进行测试：
cp example_modules/*.py .

# 3. 运行打包脚本
# Windows:
build.bat

# Linux/Mac:
chmod +x build.sh
./build.sh

# 4. 找到可执行文件
# 位置：dist/XRD_PostProcessing/XRD_PostProcessing.exe
```

### 方法2：手动打包

```bash
# 安装PyInstaller
pip install pyinstaller

# 执行打包
pyinstaller --clean xrd_app.spec

# 可执行文件在 dist 文件夹中
```

---

## 📁 项目结构

```
curve-fitting/
├── main.py                        # 主程序入口
├── xrd_app.spec                   # PyInstaller配置
├── build.bat                      # Windows打包脚本
├── build.sh                       # Linux/Mac打包脚本
├── requirements_gui.txt           # GUI应用依赖
├── .gitignore                     # Git忽略文件
│
├── resources/                     # 资源文件夹
│   ├── app_icon.ico              # 应用图标（需要您提供）
│   └── README.md                 # 资源说明
│
├── example_modules/               # 示例模块（用于测试）
│   ├── theme_module.py
│   ├── powder_module.py
│   ├── radial_module.py
│   ├── single_crystal_module.py
│   └── README.md
│
├── curve_fitting_script/          # 原有的曲线拟合脚本
│   └── curve_fitting.py
│
├── BUILD_GUIDE.md                 # 详细打包指南
├── QUICKSTART_CN.md               # 快速开始指南（中文）
└── README_GUI.md                  # 本文件
```

---

## 📚 文档导航

| 文档 | 说明 | 适合人群 |
|------|------|---------|
| [QUICKSTART_CN.md](QUICKSTART_CN.md) | 5分钟快速开始 | 想快速测试打包的用户 |
| [BUILD_GUIDE.md](BUILD_GUIDE.md) | 完整打包指南 | 需要详细说明的用户 |
| [resources/README.md](resources/README.md) | 图标准备指南 | 需要自定义图标的用户 |
| [example_modules/README.md](example_modules/README.md) | 示例模块说明 | 想先测试打包的用户 |

---

## 🎯 使用场景

### 场景1：测试打包功能

如果您想先测试打包功能是否正常：

```bash
# 1. 复制示例模块
cp example_modules/*.py .

# 2. 直接打包
build.bat  # Windows
./build.sh # Linux/Mac
```

### 场景2：打包完整应用

如果您已有完整的模块文件：

```bash
# 1. 将您的模块文件复制到项目根目录：
#    - theme_module.py
#    - powder_module.py
#    - radial_module.py
#    - single_crystal_module.py

# 2. （可选）准备图标文件
#    将 app_icon.ico 放入 resources/ 文件夹

# 3. 执行打包
build.bat  # Windows
```

---

## 🔧 配置说明

### 修改应用名称

编辑 `xrd_app.spec` 文件：

```python
exe = EXE(
    ...
    name='您的应用名称',  # 修改这里
    ...
)
```

### 添加更多依赖

编辑 `requirements_gui.txt`，添加新的包：

```
your_new_package>=1.0.0
```

然后在 `xrd_app.spec` 的 `hiddenimports` 中添加：

```python
hiddenimports=[
    'tkinter',
    'your_new_package',  # 添加这里
    ...
],
```

### 自定义图标

1. 将您的图标文件放入 `resources/app_icon.ico`
2. 如果图标不是.ico格式，使用在线工具转换：https://www.icoconverter.com/

---

## 💡 技术栈

- **GUI框架**: Tkinter
- **打包工具**: PyInstaller
- **科学计算**: NumPy, SciPy, Pandas
- **数据可视化**: Matplotlib
- **图像处理**: Pillow

---

## 🐛 常见问题

### Q: 打包后运行出错？

**A:** 检查是否所有模块都已包含在 `xrd_app.spec` 的 `hiddenimports` 中。

### Q: 图标不显示？

**A:** 确保 `resources/app_icon.ico` 存在且格式正确（必须是ICO格式）。

### Q: 文件太大？

**A:** 使用虚拟环境打包，只安装必需的依赖包。

### Q: 如何创建桌面快捷方式？

**A:**
1. 找到 `dist/XRD_PostProcessing/XRD_PostProcessing.exe`
2. 右键 → 发送到 → 桌面快捷方式

更多问题请查看 [BUILD_GUIDE.md](BUILD_GUIDE.md#常见问题)

---

## 📞 联系我们

- 📧 lixd@ihep.ac.cn
- 📧 fzhang@ihep.ac.cn
- 📧 yswang@ihep.ac.cn

---

## 📄 许可证

本项目遵循 [Apache-2.0 许可证](LICENSE)

---

## 🙏 致谢

感谢所有为XRD数据处理做出贡献的开发者和研究人员！

---

**💜 Ready to sparkle your XRD data? 🌈 开始打包吧！**

