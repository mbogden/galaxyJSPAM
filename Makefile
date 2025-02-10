all: compile_notebooks make_spam

# Compile Fortran code in Simulator directory
make_spam:
	cd Simulator;make

# Compile jupyter notebooks into python scripts in misc directories
# pip install is my short term solution.  Permaent solution required updating singularity image.
compile_notebooks:
	pip install --upgrade nbconvert 
	bash utilities/compile_notebooks.sh Simulator utilities/update_build_env.py
	bash utilities/compile_notebooks.sh IllustrisTNG utilities/update_build_env.py
	bash utilities/compile_notebooks.sh Optimization_Methods utilities/update_build_env.py