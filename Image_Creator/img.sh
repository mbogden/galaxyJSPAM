# Simple image creator

g++ `pkg-config opencv --cflags --libs` -I cpp/include -c cpp/src/imgClass.cpp -o cpp/build/imgClass.o
g++ `pkg-config opencv --cflags --libs` -I cpp/include -c cpp/src/imgCreator.cpp -o cpp/build/imgCreator.o
g++ `pkg-config opencv --cflags --libs` -I cpp/include -c cpp/src/img.cpp -o cpp/build/img.o

g++ cpp/build/imgClass.o cpp/build/imgCreator.o cpp/build/img.o -o cpp/bin/img.exe `pkg-config opencv --cflags --libs` -fopenmp  -I cpp/include 
./cpp/bin/img.exe run0000/ param0202.txt
