# 示例模块文件夹

此文件夹包含用于测试打包的示例模块文件。

## 使用说明

### 如果您已有完整的模块文件：

将您的完整模块文件（theme_module.py等）复制到项目根目录，**不需要使用**这些示例文件。

### 如果您想先测试打包功能：

将此文件夹中的文件复制到项目根目录：

```bash
# Windows
copy example_modules\*.py .

# Linux/Mac
cp example_modules/*.py .
```

## 文件说明

- **theme_module.py** - 提供GUI基础类和组件
- **powder_module.py** - 粉末XRD处理模块示例
- **radial_module.py** - 径向积分模块示例
- **single_crystal_module.py** - 单晶XRD模块示例

这些是简化的示例实现，仅用于演示界面布局和测试打包功能。

## 注意事项

⚠️ 这些示例模块**不包含**实际的数据处理逻辑，仅用于：
1. 测试应用打包流程
2. 验证界面布局
3. 学习模块结构

在生产环境中，请使用您完整实现的模块文件。
