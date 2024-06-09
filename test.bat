 setlocal

 set PYTHON_PATH=C:\Users\Yegorius\Downloads\VSCode_cpp_py\VSCode\Code.exe\..\Python\Python310\python.exe

 set SCRIPT_PATH=c:\Users\Yegorius\Desktop\hack\images\process_image.py

 set SUBSTRATE_ORIGINAL_PATH=layouts\layout_2021-08-16.tif
@REM  %PYTHON_PATH% %SCRIPT_PATH% %SUBSTRATE_ORIGINAL_PATH%

set SUBSTRATE_PATH=layouts\layout_2021-06-15.tif
 %PYTHON_PATH% %SCRIPT_PATH% %SUBSTRATE_PATH%
set SUBSTRATE_PATH=layouts\layout_2021-10-10.tif
 %PYTHON_PATH% %SCRIPT_PATH% %SUBSTRATE_PATH%
set SUBSTRATE_PATH=layouts\layout_2022-03-17.tif
 %PYTHON_PATH% %SCRIPT_PATH% %SUBSTRATE_PATH%

endlocal