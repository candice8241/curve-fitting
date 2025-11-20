@echo off
REM PyInstaller 构建脚本 - 方法二：--collect-all

echo ========================================
echo   PyInstaller 构建脚本
echo   方法：--collect-all (最简单)
echo ========================================
echo.

REM 1. 关闭可能运行的程序
echo [1/4] 关闭正在运行的程序...
taskkill /F /IM XRD_PostProcessing.exe 2>nul
timeout /t 2 /nobreak >nul

REM 2. 清理旧文件
echo [2/4] 清理旧的构建文件...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

REM 3. 开始构建（不使用 spec 文件）
echo [3/4] 开始 PyInstaller 构建...
echo.
pyinstaller main.py ^
    --collect-all pyFAI ^
    --collect-all fabio ^
    --name XRD_PostProcessing ^
    --clean ^
    --noconsole

REM 4. 检查结果
echo.
echo [4/4] 检查构建结果...
if exist "dist\XRD_PostProcessing\XRD_PostProcessing.exe" (
    echo.
    echo ========================================
    echo   ✓ 构建成功！
    echo ========================================
    echo.
    echo 程序位置：dist\XRD_PostProcessing\XRD_PostProcessing.exe
    echo.
    echo 按任意键测试运行程序...
    pause >nul
    cd dist\XRD_PostProcessing
    start XRD_PostProcessing.exe
    cd ..\..
) else (
    echo.
    echo ========================================
    echo   ✗ 构建失败！
    echo ========================================
    echo.
    echo 请检查上面的错误信息
)

echo.
pause
