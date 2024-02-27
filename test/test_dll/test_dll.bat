@REM del test_dll.obj test_dll.dll
nvcc -c test_dll.cpp &
g++ -shared test_dll.obj -o test_dll.dll -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64" -lcudart -lcuda &
python test_dll.py