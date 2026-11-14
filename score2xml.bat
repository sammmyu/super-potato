@echo off
rem Drag one or more score images (or a PDF) onto this file, or run:
rem   score2xml.bat page1.jpg [page2.jpg ...] [-o out.xml] [--title "..."] [--composer "..."]
rem S2X_CMDLINE lets score2xml.py tell a drag-and-drop launch (keep window open) from a terminal.
setlocal EnableDelayedExpansion
set "S2X_CMDLINE=!cmdcmdline!"
setlocal DisableDelayedExpansion
"%~dp0.venv\Scripts\python.exe" "%~dp0score2xml.py" %*
