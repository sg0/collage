# collage: point-to-point collective algorithm benchmark
#
#   make                       host-only build
#   make CUDA=1                with GPU buffers (--gpu); needs a CUDA-aware MPI
#   make CUDA=1 CUDA_ARCH=-arch=sm_90 CUDA_HOME=/opt/cuda
#   make NO_LARGE_COUNT=1      MPI-3 libraries without MPI_Send_c & co.
#   make CUDA=1 ST=mpich       + stream-triggered transport (--transport mpich):
#                              MPICH >= 4.1 built with CUDA (MPIX_Stream enqueue API)
#   make CUDA=1 ST=acx ACX_HOME=/path/to/mpi-acx
#                              + NVIDIA MPI-ACX (--transport acx), on any MPI with
#                              MPI_THREAD_MULTIPLE and MPI-4 partitioned calls
#   make CUDA=1 ST=cray        + Cray MPICH stream-triggered ops (--transport cray)
#   make CUDA=1 ST=ucp UCX_HOME=/opt/ucx CUDA_ARCH=-arch=sm_90
#                              + UCX GPU device API (--transport ucp): device-initiated
#                              one-sided puts; UCX >= 1.20 built --with-cuda (and
#                              with GDA for ConnectX InfiniBand/RoCE)
MPICXX    ?= mpicxx
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra

# the binary is the default target, whatever order the rules below come in
.DEFAULT_GOAL := collage
all: collage
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

gpu_reduce.o: gpu_reduce.cu gpu_ops.hpp .build-config
	$(NVCC) $(NVCCFLAGS) -c -o $@ gpu_reduce.cu
endif

ifneq ($(ST),)
ifneq ($(CUDA),1)
$(error ST=$(ST) needs CUDA=1)
endif
ifeq ($(ST),mpich)
CXXFLAGS += -DCOLLAGE_ST_MPICH
else ifeq ($(ST),acx)
ACX_HOME  ?= $(HOME)/mpi-acx
# libmpi-acx.a holds relocatable device code (nvcc -dc): device-link it once.
# The architecture must match the one MPI-ACX was built for (its NVCC_GENCODE).
ACX_ARCH  ?= $(if $(CUDA_ARCH),$(CUDA_ARCH),-arch=sm_80)
CXXFLAGS  += -DCOLLAGE_ST_ACX -I$(ACX_HOME)/include
LDFLAGS   += -L$(ACX_HOME)
OBJS      += acx_dlink.o
LIBS      := -lmpi-acx -lcudadevrt $(LIBS) -lcuda -lpthread

acx_dlink.o: .build-config
	$(NVCC) $(ACX_ARCH) -dlink -o $@ -L$(ACX_HOME) -lmpi-acx
else ifeq ($(ST),cray)
CXXFLAGS += -DCOLLAGE_ST_CRAY
else ifeq ($(ST),ucp)
UCX_HOME  ?= /usr
UCX_LIB   ?= $(UCX_HOME)/lib
CXXFLAGS  += -DCOLLAGE_ST_UCP -I$(UCX_HOME)/include
LDFLAGS   += -L$(UCX_LIB) -Wl,-rpath,$(UCX_LIB)
LIBS      += -lucp -luct -lucs
OBJS      += ucp_device.o

# the UCX device API is header-only CUDA code (GDA and CUDA-IPC lanes)
ucp_device.o: ucp_device.cu ucp_device_ops.hpp ucp_proto.hpp .build-config
	$(NVCC) $(NVCCFLAGS) -I$(UCX_HOME)/include -I. -c -o $@ ucp_device.cu
else
$(error ST must be mpich, acx, cray or ucp)
endif
endif

# Make tracks files, not flags: record the build configuration so that
# switching between `make` and `make CUDA=1` (or changing flags) relinks.
BUILD_CONFIG := $(MPICXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) $(NVCCFLAGS)
.build-config: FORCE
	@echo '$(BUILD_CONFIG)' | cmp -s - $@ || echo '$(BUILD_CONFIG)' > $@

collage: bench.cpp $(HEADERS) $(OBJS) .build-config
	$(MPICXX) $(CXXFLAGS) -o $@ bench.cpp $(OBJS) $(LDFLAGS) $(LIBS)

clean:
	rm -f collage gpu_reduce.o acx_dlink.o ucp_device.o .build-config

FORCE:
.PHONY: all clean FORCE
