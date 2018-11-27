# Makefile for the SPAM system
# Written by John Wallin
# Jun 9, 1997

# --------------------------------------------------------------------------- #
# TODO
# --------------------------------------------------------------------------- #
# - Ask Dr. Wallin about the libraries that need to be linked
# 	- How recent are these settings?
# --------------------------------------------------------------------------- #

# may have to choose the compiler and
# compilation flags appropriate for your system


## Harlie settings
#F77 = g77
#F90 = gfortran -g
#F90LIBS = -L/usr/local/pgplot_gnu/pgplot -lpgplot -lpng -lz -L/usr/X11R6/lib -lX11
#FLAG =
#
## Macintosh settings
#F77 = gfortran
#F90 = gfortran -g
#F90LIBS = -L/sw/lib/pgplot -lpgplot -L/sw/lib -lpng -lz \
#	-L/usr/X11R6/lib -lX11 -L/sw/lib -laquaterm \
#	-Wl,-framework -Wl,Foundation
#FLAG =


# skynet settings
#F77 = gfortran
#F90 = gfortran -pg -fbounds-check -Wall -fimplicit-none
#F90 = gfortran -O3
#F90LIBS = -L/usr/local/pgplot_gnu/pgplot -lpgplot -lpng -lz -L/usr/X11R6/lib -lX11
#FLAG =
#
#F90LIBS = -L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk/usr/lib

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
# F90 settings
# --------------------------------------------------------------------------- #
F90         := gfortran -pg -fbounds-check -fimplicit-none -pedantic
F90BIN      := fortran/bin
F90SRCDIR   := fortran/src
F90INCLUDES := fortran/includes
F90OBJDIR   := fortran/obj
F90OBJS     := parameters_module.o io_module.o df_module.o setup_module.o \
			   integrator.o init_module.o basic_run.o
F90BUILDDIR := fortran/build
F90LIBS =

F90MAINTARGETS := $(addprefix $(F90BIN)/,basic_run mod_run)

# --------------------------------------------------------------------------- #
# C++ settings
# --------------------------------------------------------------------------- #
CXX         := g++
CXXFLAGS      := -Wall -std=c++17
CXXBIN      := cpp/bin
CXXSRCDIR   := cpp/src
CXXINCLUDES := cpp/includes
CXXOBJDIR   := cpp/obj
CXXBUILDDIR := cpp/build
CXXLIBS      =
LDFLAGS      = `pkg-config --cflags --libs opencv`



# --------------------------------------------------------------------------- #
# Main targets
# --------------------------------------------------------------------------- #
all: setup standard fillbin
standard: basic_run mod_run


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
	@$(MKDIR) $(F90OBJDIR)
	@$(MKDIR) $(F90BUILDDIR)
	@$(MKDIR) $(F90BIN)
	@$(MKDIR) $(CXXOBJDIR)
	@$(MKDIR) $(CXXBUILDDIR)
	@$(MKDIR) $(CXXBIN)
	@$(MKDIR) $(BIN)
.PHONY: setup

clean:
	@$(RM) basic_run *.o *.mod mod_run
	@$(RM) $(F90MAINTARGETS)
	@$(RM) $(F90OBJDIR)
	@$(RM) $(F90BUILDDIR)
	@$(RM) $(CXXOBJDIR)
	@$(RM) $(CXXBUILDDIR)
	@$(RM) $(BIN)
.PHONY: clean

veryclean: clean
	@$(RM) basic_run idriver *.o *.mod process_pnm process regenerate fstates trajectory zoo_reproduce gmon.out a.* plt *.jpg *.mpg fort.* pfile mpeg_parameters
	@$(RM) $(shell find . -name \*.dSYM)
	@$(RM) a_*
	@$(RM) $(shell find . -name gscript)
.PHONY: veryclean


# --------------------------------------------------------------------------- #
# TODO
# - Figure out what this does...
# --------------------------------------------------------------------------- #
# process_pnm
####reproduce
#########

##########

.c.o:
	$(CC) $(CFLAGS) -c $<

.cc.o:
	$(CC) $(CFLAGS) -c $<

#process_pnm: process_pnm.c
#	$(CC) -o process_pnm process_pnm.c

# --------------------------------------------------------------------------- #
# F90
# --------------------------------------------------------------------------- #
# Main Targets
# ------------------------------------ #
basic_run: $(F90OBJS)
	$(ALERT)
	$(F90) -o $(F90BIN)/$@ $(F90LIBS) -g $(addprefix $(F90OBJDIR)/,$^)

mod_run: $(F90OBJS)
	$(ALERT)
	$(F90) -o $(F90BIN)/$@ $(F90LIBS) -g $(addprefix $(F90OBJDIR)/,$^)
# ------------------------------------ #

init_module.o: $(F90SRCDIR)/init_module.f90
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

parameters_module.o: $(F90SRCDIR)/parameters_module.f90
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

fitness_module.o: $(F90SRCDIR)/fitness_module.f90
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

setup_module.o: $(F90SRCDIR)/setup_module.f90 parameters_module.o
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

io_module.o: $(F90SRCDIR)/io_module.f90 parameters_module.o
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

integrator.o:  $(F90SRCDIR)/integrator.f90 parameters_module.o
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

align_module.o:  $(F90SRCDIR)/align_module.f90
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

df_module.o:  $(F90SRCDIR)/df_module.f90
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

basic_run.o: $(F90SRCDIR)/basic_run.f90 parameters_module.o
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

mod_run.o: $(F90SRCDIR)/mod_run.f90 parameters_module.o
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@


# --------------------------------------------------------------------------- #
# TODO
# - Figure out what this does...
# --------------------------------------------------------------------------- #
#align_module.o

# --------------------------------------------------------------------------- #
# C++ FIXME
# --------------------------------------------------------------------------- #
#im: $(CXXBIN)/im
#
#.PHONY: im
#all_cxx: imall diff
#
#imall: $(CXXSRCDIR)/imall.cpp  $(CXXSRCDIR)/imgCreator.cpp
#	$(CXX) $(CXXFLAGS) $^ -o $(CXXBIN)/$@ -fopenmp -ggdb $(LDFLAGS)
#
#diff: $(CXXSRCDIR)/diff.cpp $(CXXSRCDIR)/imgClass.cpp
#	$(CXX) $(CXXFLAGS) $^ -o $(CXXBIN)/$@ -ggdb $(LDFLAGS)

#diff: $(CXXSRCDIR)/diff.cpp
#	@echo "$(CXX) $(CXXFLAGS) $(LDFLAGS) -ggdb -o $(CXXBIN)/$@"
#	$(CXX) $(CFLAGS) $(LDFLAGS) -ggdb -o $(CXXBIN)/$@
#
#im: $(CXXBUILDDIR)/Galaxy_Class.o
#	$(CXX) $(CXXFLAGS) $(LDFLAGS) -ggdb -o $(CXXBIN)/$@ $<
#
#$(CXXBUILDDIR)/Galaxy_Class.o: $(CXXSRCDIR)/Galaxy_Class.cpp
#	$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c -o $@ $<

