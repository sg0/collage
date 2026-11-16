# collage: point-to-point collective algorithm benchmark
#
#   make                       host-only build
#   make CUDA=1                with GPU buffers (--gpu); needs a CUDA-aware MPI
#   make CUDA=1 CUDA_ARCH=-arch=sm_90 CUDA_HOME=/opt/cuda
#   make NO_LARGE_COUNT=1      MPI-3 libraries without MPI_Send_c & co.
MPICXX    ?= mpicxx
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra
LDFLAGS   ?=
LIBS      ?=

ifeq ($(NO_LARGE_COUNT),1)
CXXFLAGS += -DCOLLAGE_NO_LARGE_COUNT
endif

HEADERS = $(wildcard *.hpp)
OBJS    =

ifeq ($(CUDA),1)
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
NVCCFLAGS ?= -O3 -std=c++17 $(CUDA_ARCH)
CXXFLAGS  += -DCOLLAGE_CUDA -I$(CUDA_HOME)/include
LDFLAGS   += -L$(CUDA_HOME)/lib64
LIBS      += -lcudart
OBJS      += gpu_reduce.o

gpu_reduce.o: gpu_reduce.cu gpu_ops.hpp
	$(NVCC) $(NVCCFLAGS) -c -o $@ gpu_reduce.cu
endif

collage: bench.cpp $(HEADERS) $(OBJS)
	$(MPICXX) $(CXXFLAGS) -o $@ bench.cpp $(OBJS) $(LDFLAGS) $(LIBS)

clean:
	rm -f collage gpu_reduce.o

.PHONY: clean
