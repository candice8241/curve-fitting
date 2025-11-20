@echo off
REM 检查模块中是否导入了 pyFAI
REM 用法：在项目目录运行此脚本

echo ========================================
echo 检查代码中的 pyFAI 导入
echo ========================================
echo.

set "TARGET_DIR=D:\HEPS\ID31\dioptas_data\github_felicity\batch\HP_full_package"

if not exist "%TARGET_DIR%" (
    echo [警告] 目标目录不存在: %TARGET_DIR%
    echo 请检查 pathex 路径是否正确
    pause
    exit /b 1
)

echo 正在检查目录: %TARGET_DIR%
echo.

findstr /S /I /M "import pyFAI\|from pyFAI" "%TARGET_DIR%\*.py" 2>nul

if errorlevel 1 (
    echo.
    echo [✓] 未发现 pyFAI 导入
    echo 您可以安全地排除 pyFAI
) else (
    echo.
    echo [!] 发现 pyFAI 导入！
    echo.
    echo 您有两个选择：
    echo.
    echo 方案 A - 如果不需要 pyFAI 功能：
    echo   1. 在上面列出的文件中注释掉 pyFAI 导入
    echo   2. 注释掉使用 pyFAI 的代码
    echo   3. 重新打包
    echo.
    echo 方案 B - 如果需要 pyFAI 功能：
    echo   1. 在 xrd_app.spec 中从 excludes 移除 pyFAI
    echo   2. 取消注释 hiddenimports 中的 pyFAI 模块
    echo   3. 使用: pyinstaller --clean --collect-all pyFAI xrd_app.spec
)

echo.
pause
