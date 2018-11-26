g++ -Wall image_creator/imall.cpp  image_creator/imgCreator.cpp -o im.out -fopenmp -ggdb `pkg-config --cflags --libs opencv`
./im.out output/587724234257137777 image_creator/parameters/param0555.txt
