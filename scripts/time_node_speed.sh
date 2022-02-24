#source galaxyJSPAM/simr_enr/bin/activate

echo "HI!"
echo "Begin Timing" > time.txt

# Kill all of my python processes before beginning. 
pkill -9 python3

for (( i = 1; i <= 8; i=i+1 ))
do
    { echo "Nodes: $i" ; }>>time.txt 2>&1
    echo "Nodes: $i"
    
    echo "time mpirun -n $i python3 galaxyJSPAM/main_SIMR.py -targetDir galaxyJSPAM/targetDir -newScore -paramName zoo_0_direct_scores -overWrite"
    { echo "time mpirun -n $i python3 galaxyJSPAM/main_SIMR.py -targetDir galaxyJSPAM/targetDir -newScore -paramName zoo_0_direct_scores -overWrite" ; } >>time.txt 2>&1
    { time mpirun -n $i python3 galaxyJSPAM/main_SIMR.py -targetDir galaxyJSPAM/targetDir -newScore -paramName zoo_0_direct_scores -overWrite  ; } >>time.txt 2>&1
    
    pkill -9 python3
    
    echo "time python3 galaxyJSPAM/main_SIMR.py -nProc $i"
    { echo "time python3 galaxyJSPAM/main_SIMR.py -nProc $i" ; } >>time.txt 2>&1
    { time python3 galaxyJSPAM/main_SIMR.py -nProc $i -targetDir galaxyJSPAM/targetDir -newScore -paramName zoo_0_direct_scores -overWrite  ; } >>time.txt 2>&1
    
    # Clean up hanging processes after I'm done. 
    pkill -9 python3
    
    echo "time mpirun -n $i python3 galaxyJSPAM/main_SIMR.py -targetDir galaxyJSPAM/targetDir -newScore -paramName zoo_0_direct_scores -overWrite"
    { echo "time mpirun -n $i python3 galaxyJSPAM/main_SIMR.py -targetDir galaxyJSPAM/targetDir -newScore -paramName zoo_0_direct_scores -overWrite" ; } >>time.txt 2>&1
    { time mpirun -n $i python3 galaxyJSPAM/main_SIMR.py -targetDir galaxyJSPAM/targetDir -newScore -paramName zoo_0_direct_scores -newImage -overWrite  ; } >>time.txt 2>&1
    
    pkill -9 python3
    
    echo "time python3 galaxyJSPAM/main_SIMR.py -nProc $i"
    { echo "time python3 galaxyJSPAM/main_SIMR.py -nProc $i" ; } >>time.txt 2>&1
    { time python3 galaxyJSPAM/main_SIMR.py -nProc $i -targetDir galaxyJSPAM/targetDir -newScore -paramName zoo_0_direct_scores -newImage -overWrite ; } >>time.txt 2>&1
    
    # Clean up hanging processes after I'm done. 
    pkill -9 python3
     
    echo "$i Done"
    
    
done
