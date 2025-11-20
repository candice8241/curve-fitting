@echo off
REM ===================================================
REM XRD Data Post-Processing - Windows打包脚本
REM ===================================================

echo ========================================
echo XRD应用程序打包工具
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    pause
    exit /b 1
)

echo [1/5] 检查Python环境...
python --version
echo.

REM 检查并安装PyInstaller
echo [2/5] 检查PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller未安装，正在安装...
    pip install pyinstaller
) else (
    echo PyInstaller已安装
)
echo.

REM 清理之前的构建文件
echo [3/5] 清理旧的构建文件...
if exist "build" (
    echo 删除build文件夹...
    rmdir /s /q build
)
if exist "dist" (
    echo 删除dist文件夹...
    rmdir /s /q dist
)
echo.

REM 执行打包
echo [4/5] 开始打包应用程序...
echo 这可能需要几分钟时间，请耐心等待...
echo.
pyinstaller --clean xrd_app.spec

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！请检查错误信息
    pause
    exit /b 1
)

echo.
echo [5/5] 打包完成！
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
