# 资源文件夹说明

## 图标文件

请将您的应用程序图标文件放在此文件夹中。

### 需要的图标文件：

1. **app_icon.ico** - Windows应用程序图标
   - 格式：.ico
   - 建议尺寸：256x256 或包含多种尺寸（16x16, 32x32, 48x48, 256x256）
   - 用途：应用程序窗口图标和exe文件图标

### 如何创建.ico图标文件：

#### 方法1：在线转换工具
- 访问 https://www.icoconverter.com/
- 上传您的图片（PNG、JPG等）
- 选择尺寸（建议选择所有尺寸）
- 下载生成的.ico文件

#### 方法2：使用Python PIL库
```python
from PIL import Image

# 打开图片
img = Image.open('your_image.png')

# 保存为.ico格式
img.save('app_icon.ico', format='ICO', sizes=[(16,16), (32,32), (48,48), (256,256)])
```

### 当前图标位置：

如果您已经有图标文件，请将它复制到：
- `D:\HEPS\ID31\dioptas_data\github_felicity\batch\HP_full_package\ChatGPT Image.ico`

然后重命名为 `app_icon.ico` 并放到此 `resources` 文件夹中。

### 临时图标：

如果暂时没有图标文件，打包程序也能正常运行，只是不会显示自定义图标。
