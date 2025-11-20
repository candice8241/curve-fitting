# pyFAI 循环导入错误 - 完整解决方案

## 🔥 错误信息
```
ImportError: cannot import name 'splitPixelFullCSC' from 'pyFAI.integrator.load_engines'
```

## 📋 问题原因
pyFAI 使用**动态导入**机制加载扩展模块，PyInstaller 无法通过静态分析检测到这些模块。即使在 `hiddenimports` 中添加了模块名，PyInstaller 打包后 pyFAI 的动态导入仍然可能失败。

---

## ✅ 解决方案（三种方法，按推荐顺序尝试）

### 🥇 方法一：使用 Runtime Hook（最强力，推荐！）

**文件：XRD_PostProcessing_V2.spec + pyi_rth_pyFAI.py**

#### 原理
Runtime Hook 在程序启动时**强制预加载**所有 pyFAI 模块，解决动态导入问题。

#### 操作步骤

```bash
# 1. 将这两个文件复制到项目目录：
#    - XRD_PostProcessing_V2.spec
#    - pyi_rth_pyFAI.py

# 2. 确保文件结构：
你的项目/
├── main.py
├── radial_module.py
├── XRD_PostProcessing_V2.spec  ← 新的 spec 文件
└── pyi_rth_pyFAI.py           ← Runtime Hook

# 3. 运行构建命令
pyinstaller XRD_PostProcessing_V2.spec --clean

# 4. 测试
cd dist\XRD_PostProcessing
XRD_PostProcessing.exe
```

#### 关键点
- `runtime_hooks=['pyi_rth_pyFAI.py']` 在 spec 文件中指定了 runtime hook
- Runtime hook 在程序启动时立即导入所有 pyFAI.ext 模块
- 还会将模块注入到 `pyFAI.integrator.load_engines` 中

---

### 🥈 方法二：使用 --collect-all 命令（最简单）

#### 操作步骤

```bash
# 直接使用命令行参数，自动收集所有 pyFAI 相关文件
pyinstaller main.py --collect-all pyFAI --collect-all fabio --name XRD_PostProcessing --clean --noconsole
```

#### 优点
- 最简单，一行命令搞定
- 自动收集所有 pyFAI 模块、数据文件、二进制文件

#### 缺点
- 打包体积较大（包含了所有 pyFAI 文件，包括不需要的）
- 构建时间较长

---

### 🥉 方法三：使用 Hook + Spec（组合方案）

**文件：XRD_PostProcessing_FIXED.spec + hook-pyFAI.py**

#### 操作步骤

```bash
# 1. 创建 hooks 目录
mkdir hooks

# 2. 将 hook-pyFAI.py 复制到 hooks 目录
你的项目/
├── main.py
├── XRD_PostProcessing_FIXED.spec
└── hooks/
    └── hook-pyFAI.py

# 3. 运行构建
pyinstaller XRD_PostProcessing_FIXED.spec --additional-hooks-dir=hooks --clean
```

---

## 🎯 推荐流程

### 第一步：尝试方法一（Runtime Hook）
```bash
pyinstaller XRD_PostProcessing_V2.spec --clean
```

✅ **如果成功** → 完成！
❌ **如果失败** → 进入第二步

---

### 第二步：尝试方法二（--collect-all）
```bash
pyinstaller main.py --collect-all pyFAI --collect-all fabio --name XRD_PostProcessing --clean --noconsole
```

✅ **如果成功** → 完成！
❌ **如果失败** → 进入第三步

---

### 第三步：组合使用
```bash
# 方法 1 + 方法 2 组合
pyinstaller XRD_PostProcessing_V2.spec --collect-all pyFAI --clean
```

---

## 🔍 验证构建是否成功

### 1. 检查是否有 pyFAI 扩展模块

```bash
# 在 dist\XRD_PostProcessing\_internal 目录下查找：
dir /s *splitPixelFullCSC*
```

应该找到：
- `pyFAI\ext\splitPixelFullCSC.pyd` （Windows）
- 或 `pyFAI/ext/splitPixelFullCSC.so` （Linux）

### 2. 启用控制台查看错误

临时修改 spec 文件：
```python
exe = EXE(
    # ...
    console=True,  # 改为 True
)
```

重新构建后运行，可以看到详细错误信息。

### 3. 测试 pyFAI 导入

创建测试脚本 `test_pyfai.py`：
```python
import sys
print("Python executable:", sys.executable)

try:
    import pyFAI
    print("✓ pyFAI imported successfully")
    print("  Version:", pyFAI.__version__)

    from pyFAI.ext import splitPixelFullCSC
    print("✓ splitPixelFullCSC imported successfully")

    from pyFAI.integrator.load_engines import splitPixelFullCSC as spc
    print("✓ splitPixelFullCSC from load_engines imported successfully")

    print("\n🎉 All pyFAI imports successful!")
except Exception as e:
    print("❌ Error:", e)
    import traceback
    traceback.print_exc()
```

---

## 🛠️ 高级故障排查

### 问题 1：找不到 .pyd 或 .so 文件

**原因：**编译的扩展模块没有被打包

**解决：**
```bash
# 手动复制扩展模块
python -c "import pyFAI.ext; import os; print(os.path.dirname(pyFAI.ext.__file__))"
# 记下路径，然后在 spec 文件的 binaries 中添加：
binaries=[
    (r'C:\Python39\Lib\site-packages\pyFAI\ext\*.pyd', 'pyFAI/ext'),
],
```

### 问题 2：DLL 加载失败

**解决：**
```bash
pyinstaller XRD_PostProcessing_V2.spec --collect-all pyFAI --collect-dynamic-libs pyFAI --clean
```

### 问题 3：仍然有导入错误

**终极方案：**
```bash
# 组合所有选项
pyinstaller XRD_PostProcessing_V2.spec \
    --collect-all pyFAI \
    --collect-all fabio \
    --copy-metadata pyFAI \
    --recursive-copy-metadata pyFAI \
    --clean
```

---

## 📦 文件说明

| 文件名 | 用途 | 必需性 |
|--------|------|--------|
| `XRD_PostProcessing_V2.spec` | 主配置文件（含 runtime hook） | ⭐⭐⭐ 推荐 |
| `pyi_rth_pyFAI.py` | Runtime Hook，强制预加载模块 | ⭐⭐⭐ 配合 V2.spec 使用 |
| `XRD_PostProcessing_FIXED.spec` | 主配置文件（无 runtime hook） | ⭐⭐ 备选 |
| `hook-pyFAI.py` | PyInstaller Hook，自动收集模块 | ⭐ 可选 |

---

## ⚡ 快速命令参考

```bash
# 方法 1：Runtime Hook（推荐）
pyinstaller XRD_PostProcessing_V2.spec --clean

# 方法 2：--collect-all（最简单）
pyinstaller main.py --collect-all pyFAI --collect-all fabio --name XRD_PostProcessing --clean --noconsole

# 方法 3：组合（最强力）
pyinstaller XRD_PostProcessing_V2.spec --collect-all pyFAI --clean

# 测试构建（带控制台）
pyinstaller XRD_PostProcessing_V2.spec --clean --console

# 查看打包内容
pyi-archive_viewer dist\XRD_PostProcessing\XRD_PostProcessing.exe
```

---

## 🎓 技术细节

### Runtime Hook 的工作原理

1. **执行时机：**在应用程序主代码运行之前
2. **作用：**预先导入所有 pyFAI.ext 模块，避免动态导入失败
3. **注入机制：**将模块对象注入到 `pyFAI.integrator.load_engines`

### 为什么 hiddenimports 不够？

- PyInstaller 的 `hiddenimports` 只是告诉打包器要包含这些模块
- 但 pyFAI 的动态导入机制在运行时可能找不到这些模块
- Runtime Hook 确保模块在需要之前就已经加载到内存中

---

## ✅ 成功标志

运行 `dist\XRD_PostProcessing\XRD_PostProcessing.exe` 后：
- ✓ 不再出现 `ImportError: cannot import name 'splitPixelFullCSC'`
- ✓ radial_module 可以正常使用 pyFAI
- ✓ XRD 数据处理功能正常工作

---

## 📞 还有问题？

如果以上所有方法都失败：

1. **检查 Python 环境：**
   ```bash
   python -c "import pyFAI; from pyFAI.ext import splitPixelFullCSC; print('OK')"
   ```

2. **检查 PyInstaller 版本：**
   ```bash
   pyinstaller --version
   # 推荐 5.0 或更高
   ```

3. **尝试在虚拟环境中构建：**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install pyinstaller pyFAI fabio numpy scipy matplotlib ...
   pyinstaller XRD_PostProcessing_V2.spec --clean
   ```

4. **提供详细信息：**
   - Python 版本
   - PyInstaller 版本
   - pyFAI 版本
   - 完整的错误信息

---

**祝你构建成功！** 🚀
