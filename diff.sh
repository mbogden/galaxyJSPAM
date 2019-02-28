# Simple 

g++ -I cpp/include -c cpp/src/imgClass.cpp -o cpp/build/imgClass.o `pkg-config --cflags --libs opencv`
g++ -I cpp/include -c cpp/src/diff.cpp -o cpp/build/diff.o `pkg-config --cflags --libs opencv`
g++ -I cpp/include cpp/build/diff.o cpp/build/imgClass.o -o cpp/bin/diff.out `pkg-config --cflags --libs opencv`


./cpp/bin/diff.out output/588017702948962343/ targets/588017702948962343/588017702948962343.calibrated.png targets/588017702948962343/588017702948962343.info.txt
