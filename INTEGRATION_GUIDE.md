# 交互式峰拟合GUI集成指南

## 概述

增强版峰拟合GUI已成功集成到粉末XRD模块中。现在您可以在主模块中直接启动交互式峰拟合界面。

## 主要修改

### 1. 新增导入

```python
from peak_fitting_gui_enhanced import PeakFittingGUI
```

在文件顶部添加了对增强版峰拟合GUI的导入。

### 2. 新增实例变量

在 `__init__` 方法中添加了:

```python
# Track interactive fitting window
self.interactive_fitting_window = None
```

用于跟踪交互式拟合窗口的状态。

### 3. 新增按钮

在 `setup_integration_module()` 方法的按钮区域添加了新按钮:

```python
# NEW: Interactive Peak Fitting Button
SpinboxStyleButton(btns, "✨ Interactive Fitting", self.open_interactive_fitting,
                  width=180).pack(side=tk.LEFT, padx=6)
```

### 4. 新增方法

添加了 `open_interactive_fitting()` 方法,用于打开独立的交互式峰拟合窗口:

```python
def open_interactive_fitting(self):
    """
    Open the interactive peak fitting GUI in a new window
    """
    # Check if window already exists and is open
    if self.interactive_fitting_window is not None:
        try:
            if self.interactive_fitting_window.winfo_exists():
                # Bring window to front
                self.interactive_fitting_window.lift()
                self.interactive_fitting_window.focus_force()
                self.log("📊 Interactive fitting window brought to front")
                return
        except:
            pass

    # Create new toplevel window
    self.interactive_fitting_window = tk.Toplevel(self.root)
    self.interactive_fitting_window.title("Interactive Peak Fitting - Enhanced")

    # Set window size and position
    window_width = 1400
    window_height = 850
    screen_width = self.interactive_fitting_window.winfo_screenwidth()
    screen_height = self.interactive_fitting_window.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    self.interactive_fitting_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Create the peak fitting GUI inside this window
    fitting_app = PeakFittingGUI(self.interactive_fitting_window)
    fitting_app.setup_ui()

    # Log the action
    self.log("✨ Interactive peak fitting GUI opened in new window")

    # Handle window close event
    def on_closing():
        if messagebox.askokcancel("Close Interactive Fitting",
                                 "Are you sure you want to close the interactive fitting window?"):
            self.interactive_fitting_window.destroy()
            self.interactive_fitting_window = None
            self.log("📊 Interactive fitting window closed")

    self.interactive_fitting_window.protocol("WM_DELETE_WINDOW", on_closing)
```

## 使用方法

### 步骤 1: 确保文件结构正确

确保以下文件在同一目录下:
- `powder_xrd_module_with_interactive_fitting.py` (修改后的主模块)
- `peak_fitting_gui_enhanced.py` (增强版峰拟合GUI)
- 其他依赖文件 (batch_integration.py, half_auto_fitting.py, 等)

### 步骤 2: 更新导入

在您的主程序中,使用新的模块文件:

```python
from powder_xrd_module_with_interactive_fitting import PowderXRDModule
```

或者直接重命名文件,替换原来的powder_xrd_module.py。

### 步骤 3: 使用交互式拟合

1. 运行您的主程序
2. 在 "1D Integration & Peak Fitting" 模块中
3. 点击 "✨ Interactive Fitting" 按钮
4. 一个新的独立窗口将打开,包含完整的交互式峰拟合GUI

## 功能特性

### 窗口管理
- **单例模式**: 如果窗口已经打开,再次点击按钮会将窗口置于前台而不是创建新窗口
- **居中显示**: 新窗口自动在屏幕中央打开
- **独立运行**: 交互式拟合窗口独立于主窗口运行,不会阻塞主界面

### 日志记录
- 打开窗口时自动记录日志: "✨ Interactive peak fitting GUI opened in new window"
- 窗口已存在时: "📊 Interactive fitting window brought to front"
- 关闭窗口时: "📊 Interactive fitting window closed"

### 关闭确认
- 关闭窗口时会弹出确认对话框,防止意外关闭

## 完整的工作流程示例

### 场景 1: 批量处理 + 交互式精细调整

1. 使用 "🐿️ Run Integration" 进行批量积分
2. 使用 "🐻 Run Fitting" 进行批量拟合
3. 使用 "✨ Interactive Fitting" 打开交互式界面
4. 在交互式界面中加载特定数据文件
5. 手动选择峰位,精细调整拟合参数
6. 保存精细调整后的结果

### 场景 2: 纯交互式处理

1. 直接点击 "✨ Interactive Fitting"
2. 在打开的窗口中加载数据文件
3. 使用所有交互式功能:
   - 自动/手动峰识别
   - 背景选择和扣除
   - 数据平滑
   - 峰分组和拟合
   - 结果保存

## 注意事项

### 依赖关系
确保所有依赖库已安装:
```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### 文件路径
- 所有相关Python文件必须在Python的搜索路径中
- 建议将所有文件放在同一目录下

### 内存管理
- 关闭交互式拟合窗口会释放相关资源
- 建议在不使用时关闭窗口以节省内存

## 故障排查

### 问题: 点击按钮没有反应
**解决方案**: 检查是否正确导入了 `peak_fitting_gui_enhanced` 模块

### 问题: 窗口打开但显示空白
**解决方案**: 检查 `PeakFittingGUI` 类的 `setup_ui()` 方法是否被正确调用

### 问题: 导入错误
**解决方案**:
```python
# 检查文件路径
import sys
print(sys.path)

# 如果需要,添加当前目录到路径
sys.path.append('/path/to/your/files')
```

## 扩展可能

### 未来可以添加的功能:
1. **数据传递**: 从主模块直接传递数据到交互式拟合窗口
2. **结果回传**: 将交互式拟合结果传回主模块
3. **批量交互**: 在交互式界面中处理多个文件
4. **参数同步**: 主模块和交互式模块之间的参数同步

## 总结

这个集成提供了两种峰拟合方式:
- **批量自动拟合**: 快速处理大量数据
- **交互式精细拟合**: 对特定数据进行精细调整

两种方式可以结合使用,提供了灵活高效的数据处理工作流程。
