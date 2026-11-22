@echo off
cd /d "%~dp0"
py -3 -m http.server 7100 || python -m http.server 7100
