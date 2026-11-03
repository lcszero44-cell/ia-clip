@echo off
setlocal
call "%~dp0iniciar.bat" %*
exit /b %errorlevel%
