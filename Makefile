CXX      ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

# macOS workaround: some Command Line Tools installs are missing headers in
# /Library/Developer/CommandLineTools/usr/include/c++/v1. Fall back to the
# SDK's c++/v1 when the default location is broken. No-op on Linux.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    ifeq ($(wildcard /Library/Developer/CommandLineTools/usr/include/c++/v1/cstdint),)
        SDK_CXX_INC := $(shell xcrun --sdk macosx --show-sdk-path 2>/dev/null)/usr/include/c++/v1
        ifneq ($(wildcard $(SDK_CXX_INC)/cstdint),)
            CXXFLAGS += -isystem $(SDK_CXX_INC)
        endif
    endif
endif

.PHONY: all foundation solver tools test check-data clean

all: foundation

# The starting point: MST double-tree 2-approximation.
foundation: tsp_foundation.cpp
	$(CXX) $(CXXFLAGS) -o foundation tsp_foundation.cpp

# Your solver. Start with:  cp tsp_foundation.cpp solver.cpp
solver: solver.cpp
	$(CXX) $(CXXFLAGS) -o solver solver.cpp

# --- instructor / tooling targets ------------------------------------------

# The Held-Karp lower-bound tool (multi-threaded, C++20 -- instructor tooling
# is not bound by the competition rules). Needed only to make your own
# practice instances.
tools: tools/bound/bound
TOOLFLAGS := $(filter-out -std=c++17,$(CXXFLAGS)) -std=c++20 -pthread
tools/bound/bound: tools/bound/bound.cpp
	$(CXX) $(TOOLFLAGS) -o tools/bound/bound tools/bound/bound.cpp

# The reference solver (instructor only; the directory is gitignored).
tools/ref/solvers/refsolve: tools/ref/solvers/refsolve.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Regression tests for grade.py (tour verification, the failure accounting,
# the enforced time/memory/thread limits).
test: foundation
	python3 tools/test_grade.py

# Verify every committed lower bound is consistent with the shipped instance
# (recomputes the bound with the tool and checks it never exceeds the stored
# value by more than rounding) -- slow, instructor use.
check-data: tools
	python3 tools/pipeline.py check

clean:
	rm -f foundation solver tools/bound/bound
