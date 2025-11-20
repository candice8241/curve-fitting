@echo off
REM 清理构建脚本 - 解决权限错误

echo 正在关闭可能运行的程序...
taskkill /F /IM XRD_PostProcessing.exe 2>nul
timeout /t 2 /nobreak >nul

echo 正在删除旧的构建文件...
if exist "dist" (
    rmdir /s /q "dist"
    echo   删除 dist 目录
)

if exist "build" (
    rmdir /s /q "build"
    echo   删除 build 目录
)

if exist "__pycache__" (
    rmdir /s /q "__pycache__"
    echo   删除 __pycache__ 目录
)

echo.
echo ✓ 清理完成！
echo.
echo 现在可以运行构建命令了
pause
