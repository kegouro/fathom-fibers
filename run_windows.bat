@echo off
if not exist .venv\Scripts\python.exe (
  py -m venv .venv
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
set PYTHONPATH=%CD%\src
.venv\Scripts\python.exe -m fathom_fibers_quick gui %*
