# XRD GUI 模块化结构说明

## 📁 文件结构

原始的超长 `curve_fitting.py` 已被重构为以下模块化文件：

```
curve-fitting/
│
├── gui_base.py                  # 基础GUI组件类
├── powder_module.py             # 粉末XRD模块
├── radial_module.py             # 径向XRD模块
├── single_crystal_module.py     # 单晶XRD模块
├── main_gui.py                  # 主GUI窗口
│
└── curve_fitting.py             # 原始文件（可保留或简化为入口）
```

---

## 🎨 各文件说明

### 1. **gui_base.py** - 基础组件类
**功能**：
- 定义统一的颜色主题方案
- 提供通用UI组件创建方法
- 包含文件/文件夹选择对话框
- 提供成功提示对话框

**主要类**：
- `GUIBase` - 所有模块的基类

**主要方法**：
- `create_card_frame()` - 创建卡片样式框架
- `create_file_picker()` - 创建文件选择器
- `create_folder_picker()` - 创建文件夹选择器
- `create_entry()` - 创建文本输入框
- `browse_file()` - 文件浏览对话框
- `browse_folder()` - 文件夹浏览对话框
- `show_success()` - 显示成功提示

---

### 2. **powder_module.py** - 粉末XRD模块
**功能**：
- 1D积分和峰拟合
- 相变分析和体积计算
- Birch-Murnaghan状态方程拟合

**主要类**：
- `PowderXRDModule(GUIBase)` - 粉末XRD处理模块

**子模块**：
1. **Integration & Fitting** (积分与拟合)
   - 运行1D积分
   - 运行峰拟合
   - 完整流程

2. **Phase Analysis & BM Fitting** (相分析与BM拟合)
   - 分离原始峰和新峰
   - 计算晶胞体积
   - Birch-Murnaghan拟合

**主要方法**：
- `setup_integration_module()` - 设置积分模块UI
- `setup_analysis_module()` - 设置分析模块UI
- `run_integration()` - 执行积分
- `run_fitting()` - 执行峰拟合
- `run_full_pipeline()` - 执行完整流程
- `separate_peaks()` - 分离峰
- `run_phase_analysis()` - 相分析
- `run_birch_murnaghan()` - BM拟合

---

### 3. **radial_module.py** - 径向XRD模块
**功能**：
- 方位角积分
- 单扇区积分
- 多扇区预设积分

**主要类**：
- `RadialXRDModule(GUIBase)` - 径向XRD处理模块

**积分模式**：
1. **Single Sector** (单扇区)
   - 自定义起始/结束角度
   - 自定义扇区标签

2. **Multiple Sectors** (多扇区预设)
   - quadrants (四象限)
   - octants (八分区)
   - hemispheres (半球)
   - horizontal_vertical (水平/垂直)

**主要方法**：
- `update_radial_mode()` - 更新模式UI
- `run_azimuthal_integration()` - 执行方位角积分
- `_run_single_sector()` - 单扇区积分
- `_run_multiple_sectors()` - 多扇区积分

---

### 4. **single_crystal_module.py** - 单晶XRD模块
**功能**：
- 占位符模块（待开发）

**主要类**：
- `SingleCrystalModule(GUIBase)` - 单晶XRD模块

---

### 5. **main_gui.py** - 主GUI窗口
**功能**：
- 应用程序入口
- 管理主窗口
- 标签页切换
- 模块加载

**主要类**：
- `XRDProcessingGUI(GUIBase)` - 主GUI应用

**主要方法**：
- `setup_ui()` - 设置主界面
- `switch_tab()` - 切换标签页
- `main()` - 主函数入口

**辅助函数**：
- `launch_main_app()` - 启动主应用
- `show_startup_window()` - 显示启动窗口

---

## 🚀 使用方法

### 方式1：直接运行主GUI
```python
python main_gui.py
```

### 方式2：导入使用
```python
from main_gui import main

if __name__ == "__main__":
    main()
```

---

## 🔧 依赖关系

```
main_gui.py
    ├── gui_base.py
    ├── powder_module.py
    │   └── gui_base.py
    ├── radial_module.py
    │   └── gui_base.py
    └── single_crystal_module.py
        └── gui_base.py
```

**外部依赖**：
- `batch_appearance.py` - ModernButton, ModernTab, CuteSheepProgressBar
- `batch_integration.py` - BatchIntegrator
- `peak_fitting.py` - BatchFitter
- `batch_cal_volume.py` - XRayDiffractionAnalyzer
- `birch_murnaghan_batch.py` - BirchMurnaghanFitter
- `batch_azimuthal_integration.py` - AzimuthalIntegrator, get_preset_sectors

---

## 💡 优势

1. **模块化设计** - 每个功能独立文件，易于维护
2. **代码复用** - 基类提供通用方法，避免重复
3. **清晰结构** - 文件职责明确，易于理解
4. **易于扩展** - 添加新模块只需继承GUIBase
5. **降低耦合** - 模块间依赖最小化

---

## 📝 后续开发

### 添加新模块步骤：
1. 创建新模块文件（如 `new_module.py`）
2. 继承 `GUIBase` 类
3. 实现 `__init__()` 和 `setup_ui()` 方法
4. 在 `main_gui.py` 中导入并添加标签页

### 示例：
```python
# new_module.py
from gui_base import GUIBase

class NewModule(GUIBase):
    def __init__(self, parent, root):
        super().__init__()
        self.parent = parent
        self.root = root

    def setup_ui(self):
        # 实现UI逻辑
        pass
```

---

## 📧 联系方式
如有问题请联系：candicewang928@gmail.com

---

**版本**: v2.0 (模块化重构版)
**日期**: 2025-11-15
