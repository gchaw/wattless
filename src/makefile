# File:		makefile
# Author:	syrowikb
# Created:	November 29 2013

GCC = gcc
CFLAGS = -c -ggdb -Wall -Wno-unused-variable -Wno-unused-but-set-variable -lrt -O3
OPENCL = -I/usr/include/ -L/usr/lib64/ -Wl,-rpath,/usr/lib64/ -L/usr/lib/nvidia-319-updates/ -I/opt/AMDAPP/include -L/opt/AMDAPP/lib/x86_64
LDFLAGS = -lOpenCL -Wl,-rpath=/usr/local/cuda-5.5/lib64 # add -Wl,rpath=PATH option so that  PATH is searched for shared libraries -i.e. libcudart.su.5.5
CUDA_FLAGS = -L/usr/local/cuda-5.5/lib64 -lcudart -lcuda


SOURCES = aes.c \
	  expand_key.c \
	  cpu_cipher.c \
	  opencl_cipher.c

CUDA_SOURCES = cuda_cipher.cu
OBJECTS = $(SOURCES:.c=.o) $(CUDA_SOURCES:.cu=.o)
EXECUTABLE = aes


all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE) : $(OBJECTS)
	$(GCC) $(OBJECTS) $(OPENCL) $(CUDA_FLAGS) $(LDFLAGS) -o $@

# build all the object files
aes.o: aes.c
	$(GCC) $(CFLAGS) aes.c

aes.o: aes.c
	/usr/local/cuda-5.5/bin/nvcc $(CUDA_FLAGS) -m64 -arch=sm_35 -o aes.o --shared -Xcompiler -fPIC -c aes.c

expand_key.o: expand_key.c
	$(GCC) $(CFLAGS) expand_key.c

cpu_cipher.o: cpu_cipher.c
	$(GCC) $(CFLAGS) cpu_cipher.c

opencl_cipher.o: opencl_cipher.c
	$(GCC) $(CFLAGS) $(OPENCL) opencl_cipher.c

cuda_cipher.o: cuda_cipher.cu
	/usr/local/cuda-5.5/bin/nvcc $(CUDA_FLAGS) -m64 -arch=sm_35 -o cuda_cipher.o --shared -Xcompiler -fPIC -c cuda_cipher.cu
#/usr/local/cuda-5.5/bin/nvcc $(CUDA_FLAGS) -m64 -arch=sm_20 -arch=sm_35 -o cuda_cipher.o --shared -Xcompiler -fPIC -c cuda_cipher.cu

# do cleanup - remove object files and executable
clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)

