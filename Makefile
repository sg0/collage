# collage: point-to-point collective algorithm benchmark
MPICXX   ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
# set NO_LARGE_COUNT=1 to build against MPI-3 libraries without MPI_*_c calls
ifeq ($(NO_LARGE_COUNT),1)
CXXFLAGS += -DCOLLAGE_NO_LARGE_COUNT
endif

HEADERS = $(wildcard *.hpp)

collage: bench.cpp $(HEADERS)
	$(MPICXX) $(CXXFLAGS) -o $@ bench.cpp

clean:
	rm -f collage

.PHONY: clean
