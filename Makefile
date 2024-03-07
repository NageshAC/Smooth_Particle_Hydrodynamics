COMPILER_FLAGS = -std=c++20 
GPU_CC_FLAG = -arch=sm_70 
NVCC = nvcc
CC = g++

# compiler variable
SRC_FOLDER = src
CPP_SRC_FILES = $(wildcard ./${SRC_FOLDER}/*.cpp)
CUDA_SRC_FILES = $(wildcard ./${SRC_FOLDER}/*.cu)

# output variables
OUT_FOLDER = build
OUT_FILES_NAME = main
OUT_EXEC = ${OUT_FILES_NAME}.exe

TEST_FOLDER = test
CC_TEST_FLAGS = -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64" -lcudart -lcuda -shared
PY = python 

# initiaisation file setup
INIT_FOLDER = data
INIT_OUT_FILE = ${INIT_FOLDER}/init_setup
INIT_FILES = ${INIT_OUT_FILE}.cpp
INIT_OUT_EXEC = ${INIT_OUT_FILE}.exe

default: run

profile: compile
	nsys profile --stats=true ${OUT_FOLDER}/${OUT_EXEC}

run: compile
	${OUT_FOLDER}\${OUT_EXEC}

compile : ${SOURCE_FILES}
	${NVCC} ${COMPILER_FLAGS} ${GPU_CC_FLAG} ${CUDA_SRC_FILES} ${CPP_SRC_FILES} -o ./${OUT_FOLDER}\${OUT_EXEC} 

init: ${INIT_FILES}
	${CC} ${COMPILER_FLAGS} -O3 ${INIT_FILES} -o ${INIT_OUT_EXEC}
	${INIT_OUT_EXEC} data/init_setup.txt
	# del ${INIT_OUT_EXEC}

clean: 
	del "${OUT_FOLDER}"
	cls

pyInstall: 
	python.exe -m pip install --upgrade pip
	clear

unittest: dll
	python  ./test/test.py

dll:
	${NVCC} -std=c++20 -o ./build/smoothing_kernels.obj -c ./test/test_smoothing_kernels.cpp
	${CC} ${CC_TEST_FLAGS} ./build/smoothing_kernels.obj -o ./build/smoothing_kernels.dll
	clear
