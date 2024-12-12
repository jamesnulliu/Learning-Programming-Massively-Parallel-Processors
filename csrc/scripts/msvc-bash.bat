@echo off
setlocal

call csrc\scripts\msvc-setup.bat

call bash %*

endlocal