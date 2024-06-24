# Makefile for the fortran code in the SPAM system
# Written by Matthew Ogden
# 10 May 2019

# --------------------------------------------------------------------------- #
# General settings
# --------------------------------------------------------------------------- #
SHELL := /bin/sh
RM    := rm -rf
MKDIR := mkdir -p

# --------------------------------------------------------------------------- #
# Useful variables
# --------------------------------------------------------------------------- #
ALERT  = @echo "\n--------------------------------------------\nBuilding target $@...\n--------------------------------------------"

# --------------------------------------------------------------------------- #
# SPAM settings
# --------------------------------------------------------------------------- #
F90 		:= gfortran -pg -fbounds-check -fimplicit-none -pedantic -fPIC
BIN      	:= bin
F90SRCDIR 	:= fortran/src
F90INCLUDE 	:= fortran/includes
F90OBJDIR 	:= fortran/obj
F90OBJS 	:= parameters_module.o io_module.o df_module.o setup_module.o \
			   integrator.o init_module.o
F90BUILDDIR := fortran/build
#F90LIBS =

F90MAINTARGETS := $(addprefix $(F90BIN)/,basic_run mod_run orb_run many_endings_run)

# --------------------------------------------------------------------------- #
# Main targets
# --------------------------------------------------------------------------- #
all: setup standard
standard: basic_run orb_run many_endings_run

# --------------------------------------------------------------------------- #
# Utilities and PHONY
# --------------------------------------------------------------------------- #

setup:
	@$(MKDIR) $(F90OBJDIR)
	@$(MKDIR) $(F90BUILDDIR)
	@$(MKDIR) $(BIN)
	@$(MKDIR) output
.PHONY: setup

clean:
	@$(RM) basic_run *.o *.mod mod_run orb_run many_endings_run disk_now
	@$(RM) $(F90MAINTARGETS)
	@$(RM) $(F90OBJDIR)
	@$(RM) $(F90BUILDDIR)
	@$(RM) $(BIN)
.PHONY: clean


# --------------------------------------------------------------------------- #
# F90
# --------------------------------------------------------------------------- #
# Main Targets
# ------------------------------------ #
basic_run: $(F90OBJS) basic_run.o
	$(ALERT)
	$(F90) -o $(BIN)/$@ $(F90LIBS) $(addprefix $(F90OBJDIR)/,$^)
    
orb_run: $(F90OBJS) orb_run.o
	$(ALERT)
	$(F90) -o $(BIN)/$@ $(F90LIBS) $(addprefix $(F90OBJDIR)/,$^)

many_endings_run: $(F90OBJS) many_endings_run.o
	$(ALERT)
	$(F90) -o $(BIN)/$@ $(F90LIBS) $(addprefix $(F90OBJDIR)/,$^)

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

many_endings_run.o: $(F90SRCDIR)/many_endings_run.f90 parameters_module.o
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

orb_run.o: $(F90SRCDIR)/orb_run.f90 parameters_module.o
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) $(F90SRCDIR)/$*.f90 -o $(F90OBJDIR)/$@

# --------------------------------------------------------------------------- #
# Python F2PY settings
# --------------------------------------------------------------------------- #
F2PY := python -m numpy.f2py
FORTRAN_SRC := custom_runs.f90
MODULE_NAME := custom_runs

# Compile my_module.f90 and link with other modules
my_module.o: myroutines.f90 $(F90OBJDIR)/init_module.o $(F90OBJDIR)/parameters_module.o $(F90OBJDIR)/setup_module.o $(F90OBJDIR)/io_module.o $(F90OBJDIR)/integrator.o $(F90OBJDIR)/df_module.o
	$(ALERT)
	$(F90) -c -J$(F90BUILDDIR) -I$(F90BUILDDIR) myroutines.f90 -o $(F90OBJDIR)/$@

# Rule to compile the Fortran code into a Python module using f2py
$(MODULE_NAME).so: $(FORTRAN_SRC) $(F90OBJDIR)/init_module.o $(F90OBJDIR)/parameters_module.o $(F90OBJDIR)/setup_module.o $(F90OBJDIR)/io_module.o $(F90OBJDIR)/integrator.o $(F90OBJDIR)/df_module.o
	$(F2PY) -c $(FORTRAN_SRC) -m $(MODULE_NAME) -I$(F90BUILDDIR) $(F90OBJDIR)/init_module.o $(F90OBJDIR)/parameters_module.o $(F90OBJDIR)/setup_module.o $(F90OBJDIR)/io_module.o $(F90OBJDIR)/integrator.o $(F90OBJDIR)/df_module.o

# Ensure the custom module is built as part of the main targets
all: setup standard $(MODULE_NAME).so
