# 快速开始 - 集成交互式峰拟合GUI

## 🎯 核心修改

已将增强版峰拟合GUI成功集成到粉末XRD模块中,只需3步即可使用:

## ⚡ 快速使用

### 方法一: 更新现有代码

将你现有的导入语句从:
```python
from powder_xrd_module import PowderXRDModule
```

改为:
```python
from powder_xrd_module_with_interactive_fitting import PowderXRDModule
```

### 方法二: 替换文件

直接用新文件替换旧文件:
```bash
mv powder_xrd_module.py powder_xrd_module_old.py
mv powder_xrd_module_with_interactive_fitting.py powder_xrd_module.py
```

### 方法三: 运行测试程序

```bash
python test_integrated_module.py
```

## 🚀 使用步骤

1. **启动主程序**
   - 运行你的粉末XRD分析程序
   - 切换到 "1D Integration & Peak Fitting" 模块

2. **打开交互式拟合**
   - 点击 **"✨ Interactive Fitting"** 按钮
   - 新窗口将自动打开并居中显示

3. **使用交互式GUI**
   - 加载你的XRD数据文件 (.xy, .dat, .txt)
   - 使用所有增强功能:
     - ✨ 自动峰识别
     - 🖱️ 手动峰选择 (左键添加,右键删除)
     - 📊 背景拟合和扣除
     - 🔄 数据平滑
     - 🔬 高级峰拟合 (Pseudo-Voigt/Voigt)
     - 📁 文件导航 (前一个/后一个)
     - 💾 快速保存结果

## 📋 新增功能详情

### 1. 新增按钮
```
🐿️ Run Integration  |  🐻 Run Fitting  |  🦔 Full Pipeline  |  ✨ Interactive Fitting
```

### 2. 窗口管理
- **智能单例**: 重复点击会聚焦现有窗口,不会创建多个窗口
- **自动居中**: 新窗口自动在屏幕中央打开
- **关闭确认**: 防止意外关闭窗口丢失工作

### 3. 日志集成
所有操作都会记录到主程序的日志区域:
```
✨ Interactive peak fitting GUI opened in new window
📊 Interactive fitting window brought to front
📊 Interactive fitting window closed
```

## 🔧 主要代码修改

### 导入模块 (第17行)
```python
from peak_fitting_gui_enhanced import PeakFittingGUI
```

### 新增按钮 (第549行)
```python
SpinboxStyleButton(btns, "✨ Interactive Fitting",
                  self.open_interactive_fitting,
                  width=180).pack(side=tk.LEFT, padx=6)
```

### 核心方法 (第600-640行)
```python
def open_interactive_fitting(self):
    """打开交互式峰拟合GUI"""
    # 检查窗口是否已存在
    # 创建新的Toplevel窗口
    # 初始化PeakFittingGUI
    # 设置关闭事件处理
```

## 🎨 工作流程示例

### 场景1: 批量 + 精细调整
```
1. Run Integration (批量积分)
2. Run Fitting (批量拟合)
3. Interactive Fitting (选择特定文件精细调整)
4. Save Results (保存优化后的结果)
```

### 场景2: 纯交互式
```
1. Interactive Fitting (直接打开)
2. Load File (加载数据)
3. Auto Find Peaks (自动识别峰)
4. Manual Adjustment (手动调整)
5. Fit Peaks (拟合)
6. Quick Save (快速保存)
```

## 📦 依赖要求

确保已安装所有依赖:
```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

## ✅ 测试清单

- [ ] 导入模块无错误
- [ ] 点击 "✨ Interactive Fitting" 按钮
- [ ] 新窗口正常打开
- [ ] 可以加载XRD数据文件
- [ ] 峰识别和拟合功能正常
- [ ] 结果可以保存
- [ ] 关闭窗口有确认对话框
- [ ] 日志正确记录操作

## 🐛 故障排查

### 问题: ModuleNotFoundError: No module named 'peak_fitting_gui_enhanced'

**解决方案**:
```python
import sys
sys.path.append('/path/to/curve-fitting')  # 添加文件所在目录
```

### 问题: 窗口打开后显示空白

**解决方案**: 检查是否所有依赖都已正确安装
```bash
pip install --upgrade numpy matplotlib scipy scikit-learn
```

### 问题: 按钮点击无反应

**解决方案**: 检查控制台是否有错误信息,确保:
1. `peak_fitting_gui_enhanced.py` 在正确的位置
2. 所有导入都成功
3. 没有语法错误

## 📚 更多信息

详细文档请参阅: `INTEGRATION_GUIDE.md`

## 🎉 总结

现在你拥有一个强大的XRD数据分析工具,结合了:
- ⚡ 快速批量处理
- 🎨 灵活交互式调整
- 📊 专业级峰拟合
- 💾 便捷结果管理

享受你的数据分析之旅! 🚀
