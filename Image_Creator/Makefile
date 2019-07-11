# Makefile for the Image Creation of SPAM particle files
# Written by Matthew Ogden
# Jul 11, 2019

# --------------------------------------------------------------------------- #
# General settings
# --------------------------------------------------------------------------- #
SHELL := /bin/sh
RM    := rm -rf
MKDIR := mkdir -p

BIN   := bin

# --------------------------------------------------------------------------- #
# Useful variables
# --------------------------------------------------------------------------- #
ALERT  = @echo "\n--------------------------------------------\nBuilding target $@...\n--------------------------------------------"

# --------------------------------------------------------------------------- #
# C++ settings
# --------------------------------------------------------------------------- #
CXX         := g++-8
CXXFLAGS    := -Wall -std=c++17
CXXBIN      := cpp/bin
CXXSRCDIR   := cpp/src
CXXINCLUDE  := -I cpp/include
CXXOBJDIR   := cpp/obj
CXXBUILDDIR := cpp/build
CXXLIBS      =
LDFLAGS      = `pkg-config --cflags --libs opencv`


# --------------------------------------------------------------------------- #
# Main targets
# --------------------------------------------------------------------------- #
all: setup standard fillbin
standard: image


# --------------------------------------------------------------------------- #
# Utilities and PHONY
# --------------------------------------------------------------------------- #
fillbin:
	@echo "\nCopying all compiled binaries to top level bin..."
	@$(RM) $(shell find . -name \*.dSYM)
	@cp -f $(F90BIN)/* bin 2>/dev/null || :
	@cp -f $(CXXBIN)/* bin 2>/dev/null || :
.PHONY: fillbin

setup:
	@$(MKDIR) $(CXXOBJDIR)
	@$(MKDIR) $(CXXBUILDDIR)
	@$(MKDIR) $(CXXBIN)
	@$(MKDIR) $(BIN)
.PHONY: setup

clean:
	@$(RM) $(CXXOBJDIR)
	@$(RM) $(CXXBUILDDIR)
	@$(RM) $(BIN)
.PHONY: clean



#
# --------------------------------------------------------------------------- #
# C++ FIXME
# I'm not entirely sure where the errors are in the C++ portion, or how to fix
# them.
# --------------------------------------------------------------------------- #

#im: $(CXXBIN)/im
#
#.PHONY: im
#all_cxx: imall diff
#
#imall: $(CXXSRCDIR)/imall.cpp  $(CXXSRCDIR)/imgCreator.cpp
#	echo $(CXX)
#	$(CXX) $(CXXINCLUDE) $(CXXFLAGS) $^ -o $(CXXBIN)/$@ -fopenmp -ggdb $(LDFLAGS)
##
#diff: $(CXXSRCDIR)/diff.cpp $(CXXSRCDIR)/imgClass.cpp
#	$(CXX) $(CXXINCLUDE) $(CXXFLAGS) $^ -o $(CXXBIN)/$@ -ggdb $(LDFLAGS)
#
#diff: $(CXXSRCDIR)/diff.cpp
#	@echo "$(CXX) $(CXXFLAGS) $(LDFLAGS) -ggdb -o $(CXXBIN)/$@"
#	$(CXX) $(CFLAGS) $(LDFLAGS) -ggdb -o $(CXXBIN)/$@
#
#im: $(CXXBUILDDIR)/Galaxy_Class.o
#	$(CXX) $(CXXFLAGS) $(LDFLAGS) -ggdb -o $(CXXBIN)/$@ $<
#
#$(CXXBUILDDIR)/Galaxy_Class.o: $(CXXSRCDIR)/Galaxy_Class.cpp
#	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c -o $@ $<

