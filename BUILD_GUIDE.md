# 🔬 XRD数据后处理应用打包指南

本指南将帮助您将Python GUI应用打包成Windows可执行文件（.exe），制作成可放在桌面的应用程序。

---

## 📋 目录

1. [准备工作](#准备工作)
2. [项目结构](#项目结构)
3. [快速开始](#快速开始)
4. [详细步骤](#详细步骤)
5. [创建桌面快捷方式](#创建桌面快捷方式)
6. [常见问题](#常见问题)
7. [自定义配置](#自定义配置)

---

## 🎯 准备工作

### 1. 安装Python环境

确保已安装Python 3.8或更高版本：

```bash
python --version
```

### 2. 安装PyInstaller

PyInstaller是将Python程序打包成可执行文件的工具：

```bash
pip install pyinstaller
```

### 3. 安装项目依赖

确保所有依赖库已安装：

```bash
pip install tkinter numpy scipy matplotlib pandas pillow
```

**注意**：如果您的项目有 `requirements.txt` 文件，可以使用：

```bash
pip install -r requirements.txt
```

### 4. 准备图标文件

将您的应用图标（.ico格式）放入 `resources` 文件夹中，命名为 `app_icon.ico`。

如果您已有图标文件在其他位置，例如：
```
D:\HEPS\ID31\dioptas_data\github_felicity\batch\HP_full_package\ChatGPT Image.ico
```

请将其复制到项目的 `resources` 文件夹并重命名为 `app_icon.ico`。

---

## 📁 项目结构

打包前，请确保您的项目结构如下：

```
curve-fitting/
├── main.py                    # 主程序入口
├── theme_module.py            # 主题模块
├── powder_module.py           # 粉末XRD模块
├── radial_module.py           # 径向积分模块
├── single_crystal_module.py   # 单晶模块
├── xrd_app.spec              # PyInstaller配置文件
├── build.bat                 # Windows打包脚本
├── build.sh                  # Linux/Mac打包脚本
├── BUILD_GUIDE.md            # 本指南
└── resources/                # 资源文件夹
    ├── app_icon.ico          # 应用图标
    └── README.md             # 资源说明
```

**重要提示**：请确保以下模块文件存在于项目根目录：
- `theme_module.py`
- `powder_module.py`
- `radial_module.py`
- `single_crystal_module.py`

---

## 🚀 快速开始

### Windows用户

1. 双击运行 `build.bat`
2. 等待打包完成（约2-5分钟）
3. 在 `dist/XRD_PostProcessing/` 文件夹中找到 `XRD_PostProcessing.exe`

### Linux/Mac用户

1. 打开终端，进入项目目录
2. 添加执行权限：
   ```bash
   chmod +x build.sh
   ```
3. 运行打包脚本：
   ```bash
   ./build.sh
   ```
4. 在 `dist/XRD_PostProcessing/` 文件夹中找到可执行文件

---

## 📝 详细步骤

### 步骤1：检查依赖模块

确保所有必需的Python模块都已创建并放在项目目录中：

```bash
# 检查文件是否存在
dir main.py theme_module.py powder_module.py radial_module.py single_crystal_module.py
```

### 步骤2：准备图标（可选）

如果您想要自定义应用图标：

1. 准备一张图片（PNG、JPG等格式）
2. 使用在线工具转换为.ico格式：
   - 访问 https://www.icoconverter.com/
   - 上传图片，选择多种尺寸
   - 下载.ico文件
3. 将文件重命名为 `app_icon.ico` 并放入 `resources/` 文件夹

### 步骤3：执行打包

#### 方法A：使用打包脚本（推荐）

**Windows：**
```bash
build.bat
```

**Linux/Mac：**
```bash
chmod +x build.sh
./build.sh
```

#### 方法B：手动执行PyInstaller

```bash
# 清理旧文件
rmdir /s /q build dist  # Windows
rm -rf build dist       # Linux/Mac

# 执行打包
pyinstaller --clean xrd_app.spec
```

### 步骤4：测试应用

打包完成后：

1. 进入 `dist/XRD_PostProcessing/` 文件夹
2. 双击运行 `XRD_PostProcessing.exe`（Windows）或 `XRD_PostProcessing`（Linux/Mac）
3. 测试所有功能是否正常

---

## 🖥️ 创建桌面快捷方式

### Windows方法1：右键创建快捷方式

1. 找到 `dist/XRD_PostProcessing/XRD_PostProcessing.exe`
2. 右键点击 → 发送到 → 桌面快捷方式

### Windows方法2：手动创建快捷方式

1. 在桌面右键 → 新建 → 快捷方式
2. 浏览并选择 `XRD_PostProcessing.exe`
3. 命名快捷方式（例如："XRD数据处理"）
4. 完成

### 自定义快捷方式图标

1. 右键点击桌面快捷方式 → 属性
2. 点击"更改图标"
3. 浏览并选择 `resources/app_icon.ico`
4. 确定

---

## ❓ 常见问题

### Q1: 打包后程序无法运行？

**解决方案：**
- 确保所有依赖模块都已安装
- 检查 `xrd_app.spec` 中的 `hiddenimports` 是否包含所有必需模块
- 尝试在命令行运行查看错误信息：
  ```bash
  dist\XRD_PostProcessing\XRD_PostProcessing.exe
  ```

### Q2: 找不到某个模块？

**解决方案：**
编辑 `xrd_app.spec` 文件，在 `hiddenimports` 列表中添加缺失的模块：

```python
hiddenimports=[
    'tkinter',
    'your_missing_module',  # 添加这里
    ...
],
```

然后重新打包。

### Q3: 图标没有显示？

**解决方案：**
- 确保 `resources/app_icon.ico` 文件存在
- 检查.ico文件格式是否正确（必须是ICO格式，不能是重命名的PNG）
- 重新打包

### Q4: 打包文件太大？

**解决方案：**
- 使用虚拟环境，只安装必需的依赖
- 在 `xrd_app.spec` 中添加 `excludes` 排除不需要的模块：
  ```python
  excludes=['matplotlib.tests', 'numpy.tests', ...],
  ```

### Q5: 程序启动时出现控制台窗口？

**解决方案：**
在 `xrd_app.spec` 中确保 `console=False`：
```python
exe = EXE(
    ...
    console=False,  # 设置为False
    ...
)
```

### Q6: 缺少theme_module等模块文件？

**解决方案：**
如果您还没有创建这些模块文件，需要：
1. 创建空的模块文件作为占位符
2. 或者从 `xrd_app.spec` 和 `main.py` 中移除这些导入

---

## ⚙️ 自定义配置

### 修改应用名称

编辑 `xrd_app.spec` 文件：

```python
exe = EXE(
    ...
    name='您的应用名称',  # 修改这里
    ...
)
```

### 添加更多资源文件

如果您有其他资源文件（如图片、配置文件等），编辑 `xrd_app.spec`：

```python
added_files = [
    ('resources', 'resources'),
    ('data', 'data'),           # 添加data文件夹
    ('config.json', '.'),       # 添加配置文件
]
```

### 单文件打包

如果想打包成单个exe文件（启动会慢一些），修改 `xrd_app.spec`：

```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,        # 移到这里
    a.zipfiles,        # 移到这里
    a.datas,           # 移到这里
    [],
    name='XRD_PostProcessing',
    ...
    onefile=True,      # 添加这一行
)

# 删除或注释掉 COLLECT 部分
```

---

## 📦 分发应用

打包完成后，您可以：

### 方法1：直接分发文件夹

将整个 `dist/XRD_PostProcessing/` 文件夹压缩成ZIP文件，发送给其他用户。

用户只需：
1. 解压ZIP文件
2. 双击运行 `XRD_PostProcessing.exe`

### 方法2：创建安装程序（高级）

使用Inno Setup或NSIS等工具创建专业的安装程序：

**使用Inno Setup：**
1. 下载并安装 Inno Setup (https://jrsoftware.org/isinfo.php)
2. 创建安装脚本
3. 生成安装程序.exe

---

## 🎨 优化建议

### 1. 减小文件大小

使用虚拟环境：
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 只安装必需的包
pip install pyinstaller tkinter numpy scipy matplotlib pandas
```

### 2. 加快启动速度

- 延迟导入大型库
- 使用懒加载
- 减少启动时的初始化操作

### 3. 添加版本信息

创建 `version_info.txt`：
```
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    ...
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'Your Company'),
        StringStruct('FileDescription', 'XRD Data Post-Processing'),
        StringStruct('FileVersion', '1.0.0.0'),
        StringStruct('ProductName', 'XRD Processor'),
        StringStruct('ProductVersion', '1.0.0.0')])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
```

然后在 `xrd_app.spec` 中添加：
```python
exe = EXE(
    ...
    version='version_info.txt',
    ...
)
```

---

## 📞 技术支持

如有问题，请联系：
- lixd@ihep.ac.cn
- fzhang@ihep.ac.cn
- yswang@ihep.ac.cn

---

## 📄 许可证

本项目遵循 Apache-2.0 许可证。

---

**祝您打包顺利！💜✨**
