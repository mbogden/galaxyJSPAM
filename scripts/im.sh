g++ -Wall image_creator/im.cpp image_creator/Galaxy_Class.cpp -o im.out -fopenmp -ggdb `pkg-config --cflags --libs opencv`
./im.out output/587722984435351614/run0001 image_creator/param0001.txt
