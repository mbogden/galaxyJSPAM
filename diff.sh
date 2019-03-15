# Simple 

g++ -I cpp/include -c cpp/src/imgClass.cpp -o cpp/build/imgClass.o `pkg-config --cflags --libs opencv`
g++ -I cpp/include -c cpp/src/diffAll.cpp -o cpp/build/diffAll.o `pkg-config --cflags --libs opencv`
g++ -I cpp/include cpp/build/diffAll.o cpp/build/imgClass.o -o cpp/bin/diffAll.out `pkg-config --cflags --libs opencv`


./cpp/bin/diffAll.out output/588017702948962343/ targets/588017702948962343/588017702948962343.calibrated.png targets/588017702948962343/588017702948962343.info.txt
