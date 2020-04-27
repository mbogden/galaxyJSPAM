# Simple image creator

g++ -Wall -g -std=c++17 `pkg-config --cflags --libs opencv` -I cpp/include -c cpp/src/imgClass.cpp -o cpp/build/imgClass.o
g++ -Wall -g -std=c++17 `pkg-config --cflags --libs opencv` -fopenmp -I cpp/include -c cpp/src/imgCreator.cpp -o cpp/build/imgCreator.o
g++ -Wall -g -std=c++17 `pkg-config --cflags --libs opencv` -fopenmp -I cpp/include -c cpp/src/img.cpp -o cpp/build/img.o
g++ -Wall -g -std=c++17 `pkg-config --cflags --libs opencv` -fopenmp  -I cpp/include cpp/build/imgClass.o cpp/build/imgCreator.o cpp/build/img.o -o cpp/bin/img
./cpp/bin/img output/588017702948962343/run0000/ param0202.txt
