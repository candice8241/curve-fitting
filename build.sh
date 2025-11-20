#!/bin/bash
# ===================================================
# XRD Data Post-Processing - Linux/Mac打包脚本
# ===================================================

echo "========================================"
echo "XRD应用程序打包工具"
echo "========================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python3，请先安装Python 3.8或更高版本"
    exit 1
fi

echo "[1/5] 检查Python环境..."
python3 --version
echo ""

# 检查并安装PyInstaller
echo "[2/5] 检查PyInstaller..."
if ! python3 -m pip show pyinstaller &> /dev/null; then
    echo "PyInstaller未安装，正在安装..."
    python3 -m pip install pyinstaller
else
    echo "PyInstaller已安装"
fi
echo ""

# 清理之前的构建文件
echo "[3/5] 清理旧的构建文件..."
if [ -d "build" ]; then
    echo "删除build文件夹..."
    rm -rf build
fi
if [ -d "dist" ]; then
    echo "删除dist文件夹..."
    rm -rf dist
fi
echo ""

# 执行打包
echo "[4/5] 开始打包应用程序..."
echo "这可能需要几分钟时间，请耐心等待..."
echo ""
python3 -m PyInstaller --clean xrd_app.spec

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 打包失败！请检查错误信息"
    exit 1
fi

echo ""
echo "[5/5] 打包完成！"
echo ""
echo "========================================"
echo "打包成功！"
echo "========================================"
echo ""
echo "可执行文件位置："
echo "  dist/XRD_PostProcessing/XRD_PostProcessing"
echo ""
echo "您可以："
echo "  1. 将整个 dist/XRD_PostProcessing 文件夹复制到任何位置"
echo "  2. 运行 ./dist/XRD_PostProcessing/XRD_PostProcessing"
echo ""

# 添加执行权限
chmod +x dist/XRD_PostProcessing/XRD_PostProcessing

echo "已添加执行权限"
echo ""
