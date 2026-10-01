@echo off
rem Configure + build with clang-cl inside a VS developer environment.
rem Usage: build.cmd [extra cmake args]
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=C:\Program Files\LLVM\bin;C:\Program Files\CMake\bin;%PATH%"
if not exist build\build.ninja (
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang-cl -DSK_BUILD_TORCH=OFF -DSK_BUILD_PYTHON=OFF %* || exit /b 1
)
cmake --build build || exit /b 1
