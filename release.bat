@echo off
setlocal enabledelayedexpansion

echo Starting release build process...

REM Attempt to set up MSVC environment
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "!VS_PATH!" (
    echo Found VS Build Tools at !VS_PATH!
    call "!VS_PATH!"
    goto :env_set
)

set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "!VS_PATH!" (
    echo Found VS Community at !VS_PATH!
    call "!VS_PATH!"
    goto :env_set
)

echo WARNING: Could not find vcvars64.bat. Build might fail if MSVC environment is not set.

:env_set

REM Activate Conda
call conda activate AD
if %errorlevel% neq 0 (
    echo Conda environment activation failed or conda not found.
    echo Proceeding with current python environment...
)

REM Clean up previous builds
echo Cleaning up previous build artifacts...
if exist "build" ( rd /s /q build )
if exist "dist" ( rd /s /q dist )
if exist "orionvis.egg-info" ( rd /s /q orionvis.egg-info )
if exist "orionvis\*.egg-info" ( rd /s /q orionvis\*.egg-info )

REM Clean up compiled extensions in specific source directories (to ensure clean build)
echo Cleaning up old compiled extensions...
if exist "orionvis\extentions\inplace_abn" (
    del /q "orionvis\extentions\inplace_abn\*.pyd" "orionvis\extentions\inplace_abn\*.lib" "orionvis\extentions\inplace_abn\*.exp" >nul 2>&1
    if exist "orionvis\extentions\inplace_abn\build" ( rd /s /q "orionvis\extentions\inplace_abn\build" )
    if exist "orionvis\extentions\inplace_abn\inplace_abn.egg-info" ( rd /s /q "orionvis\extentions\inplace_abn\inplace_abn.egg-info" )
)
if exist "orionvis\extentions\selective_scan" (
    del /q "orionvis\extentions\selective_scan\*.pyd" "orionvis\extentions\selective_scan\*.lib" "orionvis\extentions\selective_scan\*.exp" >nul 2>&1
    if exist "orionvis\extentions\selective_scan\build" ( rd /s /q "orionvis\extentions\selective_scan\build" )
    if exist "orionvis\extentions\selective_scan\*.egg-info" ( rd /s /q "orionvis\extentions\selective_scan\*.egg-info" )
)

REM Build wheel package
echo Building wheel package...
python setup.py bdist_wheel
if %errorlevel% neq 0 (
    echo Wheel build failed!
    exit /b %errorlevel%
)

echo Release build successful!
echo Package is located in dist/
dir dist\*.whl

endlocal
