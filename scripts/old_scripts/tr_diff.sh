g++ -ggdb difference_code/diff.cpp -o diff.out `pkg-config --cflags --libs opencv`
gdb --args ./diff.out output/587722984435351614/ output/587722984435351614/target_images/587722984435351614.calibrated.png
