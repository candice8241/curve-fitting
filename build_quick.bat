@echo off
REM ===================================================
REM 快速打包脚本 - 自动收集 pyFAI
REM ===================================================

echo ========================================
echo XRD 快速打包（包含 pyFAI）
echo ========================================
echo.

cd /d "%~dp0"

REM 清理旧文件
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

echo 正在打包（自动收集 pyFAI）...
echo.

REM 直接执行打包，使用 --collect-all pyFAI
pyinstaller --clean --collect-all pyFAI xrd_app.spec

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 打包成功！
echo ========================================
echo.
echo 可执行文件: dist\XRD_PostProcessing\XRD_PostProcessing.exe
echo.

pause
