# 🔧 故障排除指南 - Troubleshooting

当您遇到打包问题时，请参考本指南。

---

## ❌ 错误：TypeError: unhashable type: 'dict'

### 错误信息
```
TypeError: unhashable type: 'dict'
```

### 原因分析
这是 PyInstaller 配置文件中的兼容性问题，通常由以下原因造成：

1. **spec 文件中使用了 `hooksconfig={}` 参数**（在某些 PyInstaller 版本中不兼容）
2. PyInstaller 版本与配置语法不匹配

### 解决方案

#### ✅ 已修复！
最新版本的 `xrd_app.spec` 文件已经修复了这个问题。请执行以下操作：

**方案1：拉取最新代码（推荐）**

```bash
# 拉取修复后的代码
git pull origin claude/package-gui-exe-012umeduw8bYoxoZDCX6Ukx4

# 重新打包
build.bat  # Windows
./build.sh # Linux/Mac
```

**方案2：手动修复（如果无法拉取代码）**

编辑 `xrd_app.spec` 文件：

1. 在文件顶部添加：
```python
import os
```

2. 删除 `hooksconfig={}` 这一行：
```python
a = Analysis(
    ...
    hookspath=[],
    # hooksconfig={},  ← 删除或注释掉这一行
    runtime_hooks=[],
    ...
)
```

3. 修改图标路径部分（使其成为可选）：
```python
# 在 exe = EXE(...) 之前添加
icon_path = 'resources/app_icon.ico' if os.path.exists('resources/app_icon.ico') else None

exe = EXE(
    ...
    icon=icon_path,  # 使用变量而不是直接写路径
    ...
)
```

4. 保存文件并重新打包：
```bash
pyinstaller --clean xrd_app.spec
```

---

## ❌ 错误：ModuleNotFoundError: No module named 'matplotlib' (或其他科学库)

### 错误信息
```
ModuleNotFoundError: No module named 'matplotlib'
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'scipy'
```

### 原因分析
这些是可选的科学计算库。如果您的代码中实际使用了这些库，需要安装它们。但如果只是测试打包功能，可以跳过这些库。

### 解决方案

#### 方案1：使用示例模块测试打包（推荐用于快速测试）✨

示例模块**不依赖**科学计算库，可以快速测试打包功能：

```bash
# 1. 安装最小依赖（只需要 PyInstaller）
pip install pyinstaller

# 2. 复制示例模块到项目根目录
# Windows:
copy example_modules\*.py .

# Linux/Mac:
cp example_modules/*.py .

# 3. 运行打包（spec 文件已配置为不需要这些库）
build.bat  # Windows
./build.sh # Linux/Mac
```

最新的 `xrd_app.spec` 已经将这些库注释掉，不会尝试导入它们。

#### 方案2：安装完整依赖（用于完整功能）

如果您的实际模块需要这些库：

```bash
# 安装所有科学计算依赖
pip install -r requirements_gui.txt

# 这将安装：
# - numpy
# - scipy
# - pandas
# - matplotlib
# - Pillow
# - pyinstaller
```

然后在 `xrd_app.spec` 中取消注释需要的库：

```python
hiddenimports=[
    'tkinter',
    'tkinter.ttk',
    'tkinter.font',
    'theme_module',
    'powder_module',
    'radial_module',
    'single_crystal_module',
    'numpy',        # 取消注释
    'scipy',        # 取消注释
    'matplotlib',   # 取消注释
    'PIL',          # 取消注释
    'pandas',       # 取消注释
],
```

#### 方案3：选择性安装需要的库

只安装您实际使用的库：

```bash
# 只安装 numpy
pip install numpy

# 或只安装 matplotlib
pip install matplotlib

# 然后在 xrd_app.spec 中只取消注释对应的库
```

### 快速测试流程

如果您只想快速测试打包是否正常工作：

```bash
# 1. 只安装 PyInstaller
pip install pyinstaller

# 2. 使用示例模块
copy example_modules\*.py .

# 3. 确认 xrd_app.spec 中科学库已注释（最新版本已注释）

# 4. 打包
build.bat
```

这样可以在 30 秒内完成测试打包！

---

## ❌ 错误：找不到 xrd_app.spec 文件

### 错误信息
```
ERROR: Spec file "xrd_app.spec" not found!
```

### 原因分析
这个错误通常由以下原因造成：

1. **在错误的目录下运行命令**（最常见）
2. 文件没有从Git仓库正确拉取
3. 文件被意外删除

### 解决方案

#### 方案1：确保在正确的目录（推荐）

**Windows 用户：**

```batch
# 1. 使用文件资源管理器导航到项目文件夹
#    路径示例：C:\Users\YourName\curve-fitting

# 2. 在文件夹中找到 build.bat 文件

# 3. 直接双击 build.bat 运行
#    （不要在命令提示符中运行）
```

**或者在命令提示符中：**

```batch
# 1. 切换到项目目录
cd /d D:\path\to\curve-fitting

# 2. 检查文件是否存在
dir xrd_app.spec
dir main.py

# 3. 如果文件存在，运行打包脚本
build.bat
```

**Linux/Mac 用户：**

```bash
# 1. 切换到项目目录
cd /path/to/curve-fitting

# 2. 检查文件是否存在
ls -la xrd_app.spec
ls -la main.py

# 3. 如果文件存在，运行打包脚本
chmod +x build.sh
./build.sh
```

#### 方案2：从Git拉取最新代码

如果文件不存在，可能需要拉取最新代码：

```bash
# 切换到项目目录
cd curve-fitting

# 拉取最新代码
git pull origin claude/package-gui-exe-012umeduw8bYoxoZDCX6Ukx4

# 检查文件是否存在
ls -la xrd_app.spec
```

#### 方案3：手动运行PyInstaller

如果您想跳过脚本直接打包：

```bash
# 1. 确保在项目根目录
cd /path/to/curve-fitting

# 2. 检查文件列表
dir  # Windows
ls   # Linux/Mac

# 3. 手动运行PyInstaller
pyinstaller --clean xrd_app.spec
```

---

## ❌ 错误：找不到 Python

### 错误信息
```
'python' 不是内部或外部命令，也不是可运行的程序或批处理文件。
```

### 解决方案

#### Windows:

1. **检查Python是否已安装：**
   - 打开命令提示符
   - 输入：`python --version` 或 `py --version`

2. **如果未安装Python：**
   - 访问 https://www.python.org/downloads/
   - 下载 Python 3.8 或更高版本
   - **重要**：安装时勾选 "Add Python to PATH"

3. **如果已安装但无法识别：**
   - 使用 `py` 代替 `python`：
     ```batch
     py -m pip install pyinstaller
     py -m PyInstaller --clean xrd_app.spec
     ```

---

## ❌ 错误：找不到 PyInstaller

### 错误信息
```
No module named 'PyInstaller'
```

### 解决方案

```bash
# 安装 PyInstaller
pip install pyinstaller

# 或使用 py -m（Windows）
py -m pip install pyinstaller

# 验证安装
pyinstaller --version
```

---

## ❌ 错误：找不到模块（theme_module等）

### 错误信息
```
ModuleNotFoundError: No module named 'theme_module'
ImportError: cannot import name 'GUIBase'
```

### 解决方案

#### 方案1：使用示例模块测试

```bash
# 复制示例模块到项目根目录
# Windows:
copy example_modules\*.py .

# Linux/Mac:
cp example_modules/*.py .

# 然后重新打包
build.bat
```

#### 方案2：使用您自己的完整模块

确保以下文件在项目根目录：
- `theme_module.py`
- `powder_module.py`
- `radial_module.py`
- `single_crystal_module.py`

```bash
# 检查文件是否存在
dir *.py  # Windows
ls *.py   # Linux/Mac
```

---

## ❌ 错误：打包后无法运行

### 症状
- 双击exe文件没有反应
- exe文件闪退
- 显示错误信息后关闭

### 解决方案

#### 1. 在命令行运行查看错误信息

```batch
# Windows
cd dist\XRD_PostProcessing
XRD_PostProcessing.exe

# 这样可以看到完整的错误信息
```

#### 2. 检查是否缺少依赖

编辑 `xrd_app.spec`，在 `hiddenimports` 中添加缺失的模块：

```python
hiddenimports=[
    'tkinter',
    'tkinter.ttk',
    'tkinter.font',
    'numpy',
    'scipy',
    'matplotlib',
    'pandas',
    'PIL',
    # 添加您发现缺失的模块
    'missing_module_name',
],
```

然后重新打包：
```bash
pyinstaller --clean xrd_app.spec
```

#### 3. 启用控制台查看调试信息

编辑 `xrd_app.spec`：

```python
exe = EXE(
    ...
    console=True,  # 改为 True 以显示控制台
    ...
)
```

---

## ❌ 错误：图标不显示

### 症状
- exe文件显示默认Python图标
- 窗口没有自定义图标

### 解决方案

#### 1. 检查图标文件

```bash
# 确保图标文件存在
dir resources\app_icon.ico  # Windows
ls resources/app_icon.ico   # Linux/Mac
```

#### 2. 图标格式必须是 .ico

如果您的图标是PNG或JPG格式：

```python
# 使用Python转换
from PIL import Image

img = Image.open('your_image.png')
img.save('resources/app_icon.ico', format='ICO',
         sizes=[(16,16), (32,32), (48,48), (256,256)])
```

或使用在线工具：https://www.icoconverter.com/

#### 3. 临时解决：不使用图标

如果暂时不需要图标，可以注释掉相关代码：

编辑 `xrd_app.spec`：

```python
exe = EXE(
    ...
    # icon='resources/app_icon.ico',  # 注释掉这行
    ...
)
```

---

## ❌ 打包文件太大

### 症状
- dist文件夹超过500MB
- exe文件启动很慢

### 解决方案

#### 1. 使用虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 只安装必需的包
pip install pyinstaller tkinter numpy scipy matplotlib pandas

# 在虚拟环境中打包
pyinstaller --clean xrd_app.spec
```

#### 2. 排除不需要的模块

编辑 `xrd_app.spec`：

```python
a = Analysis(
    ...
    excludes=[
        'matplotlib.tests',
        'numpy.tests',
        'scipy.tests',
        'pandas.tests',
        'IPython',
        'jupyter',
    ],
    ...
)
```

---

## 📝 快速诊断清单

运行此清单来快速诊断问题：

```batch
REM Windows 快速诊断
echo 1. 检查Python版本
python --version

echo 2. 检查PyInstaller
pip show pyinstaller

echo 3. 检查当前目录
cd

echo 4. 列出项目文件
dir

echo 5. 检查必需文件
dir xrd_app.spec
dir main.py
dir example_modules

echo 6. 检查资源文件夹
dir resources
```

---

## 🆘 仍然无法解决？

### 提供详细信息

如果以上方案都不能解决您的问题，请收集以下信息：

1. **系统信息：**
   ```bash
   python --version
   pip --version
   pyinstaller --version
   ```

2. **当前目录：**
   ```bash
   cd  # Windows
   pwd # Linux/Mac
   ```

3. **文件列表：**
   ```bash
   dir  # Windows
   ls -la  # Linux/Mac
   ```

4. **完整错误信息：**
   - 复制命令行中的完整错误输出

5. **打包命令：**
   - 您执行的完整命令

### 联系支持

📧 将以上信息发送至：
- lixd@ihep.ac.cn
- fzhang@ihep.ac.cn
- yswang@ihep.ac.cn

---

## ✅ 成功打包检查清单

打包成功的标志：

- [x] `dist/XRD_PostProcessing/` 文件夹已创建
- [x] `XRD_PostProcessing.exe` 文件存在
- [x] 双击exe文件能正常启动
- [x] 界面显示正常
- [x] 所有功能可用

---

**记住：大多数问题都是因为在错误的目录下运行命令！** 🎯

**解决方案：直接双击 build.bat 文件，不要在命令行中运行。** 💡
