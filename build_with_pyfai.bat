@echo off
REM ===================================================
REM XRD Data Post-Processing - Windows打包脚本（包含pyFAI）
REM ===================================================

echo ========================================
echo XRD应用程序打包工具 (with pyFAI)
echo ========================================
echo.

REM 获取脚本所在目录并切换到该目录
cd /d "%~dp0"
echo 当前工作目录: %CD%
echo.

REM 检查关键文件是否存在
if not exist "xrd_app.spec" (
    echo [错误] 找不到 xrd_app.spec 文件！
    echo 请确保在项目根目录运行此脚本。
    echo 当前目录: %CD%
    dir /b *.spec
    pause
    exit /b 1
)

if not exist "main.py" (
    echo [错误] 找不到 main.py 文件！
    echo 请确保在项目根目录运行此脚本。
    pause
    exit /b 1
)

echo [检查] 找到 xrd_app.spec 和 main.py 文件
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo [1/6] 检查Python环境...
python --version
echo.

REM 检查并安装PyInstaller
echo [2/6] 检查PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller未安装，正在安装...
    pip install pyinstaller
) else (
    echo PyInstaller已安装
)
echo.

REM 跳过 pyFAI 检查（使用 --collect-all 会自动处理）
echo [3/6] 跳过 pyFAI 检查...
echo 注意: 使用 --collect-all pyFAI 选项会自动收集 pyFAI
echo 如果 pyFAI 未安装，打包时会自动跳过
echo.

REM 清理之前的构建文件
echo [4/6] 清理旧的构建文件...
if exist "build" (
    echo 删除build文件夹...
    rmdir /s /q build
)
if exist "dist" (
    echo 删除dist文件夹...
    rmdir /s /q dist
)
echo.

REM 执行打包（使用 --collect-all pyFAI）
echo [5/6] 开始打包应用程序（包含 pyFAI 自动收集）...
echo 这可能需要几分钟时间，请耐心等待...
echo.
echo 使用命令: pyinstaller --clean --collect-all pyFAI xrd_app.spec
echo.
pyinstaller --clean --collect-all pyFAI xrd_app.spec

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！请检查错误信息
    echo.
    echo 常见问题：
    echo 1. pyFAI 未安装：pip install pyFAI
    echo 2. 缺少依赖：pip install -r requirements_gui.txt
    echo 3. 模块冲突：删除 build 和 dist 文件夹后重试
    pause
    exit /b 1
)

echo.
echo [6/6] 打包完成！
echo.
echo ========================================
echo 打包成功！
echo ========================================
echo.
echo 可执行文件位置：
echo   dist\XRD_PostProcessing\XRD_PostProcessing.exe
echo.
echo 您可以：
echo   1. 将整个 dist\XRD_PostProcessing 文件夹复制到任何位置
echo   2. 创建桌面快捷方式指向 XRD_PostProcessing.exe
echo   3. 双击运行 XRD_PostProcessing.exe
echo.

REM 询问是否打开输出文件夹
echo 是否打开输出文件夹？ (Y/N)
set /p open_folder=
if /i "%open_folder%"=="Y" (
    explorer dist\XRD_PostProcessing
)

pause
