@echo off
rem Opens the Score to MusicXML window. You can also drop images or PDFs onto this file.
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0score2xml_gui.py" %*
) else (
  start "" pythonw "%~dp0score2xml_gui.py" %*
)
