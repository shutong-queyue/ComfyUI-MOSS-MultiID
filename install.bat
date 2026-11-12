@echo off
chcp 65001 >nul
rem 双击即可：先装 Python 包（requirements.txt），再补权重（install.py）
setlocal
set HERE=%~dp0
set PY=

for %%P in (
  "%HERE%..\..\python\python.exe"
  "%HERE%..\..\python_embeded\python.exe"
  "%HERE%..\..\..\python\python.exe"
) do if exist %%P set PY=%%~P

if "%PY%"=="" (
  where python >nul 2>nul && set PY=python
)

if "%PY%"=="" (
  echo 没找到 Python。如果你用的是自带 Python 的整合包，请把本文件放在 ComfyUI\custom_nodes\插件文件夹里再双击。
  pause
  exit /b 1
)

echo 使用 Python: %PY%
echo.
echo 〔1/2〕安装 Python 依赖（insightface / onnxruntime / opencv-python）
"%PY%" -m pip install -r "%HERE%requirements.txt"
echo.
echo 〔2/2〕补齐权重与自动切遮罩用的库（首次约 309MB）
"%PY%" "%HERE%install.py"
echo.
echo 全部完成。请重启 ComfyUI 让依赖生效。
pause
