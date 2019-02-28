# Simple 

g++ -I cpp/include -c cpp/src/imgClass.cpp -o cpp/build/imgClass.o `pkg-config --cflags --libs opencv`
g++ -I cpp/include -c cpp/src/dirDiff.cpp -o cpp/build/dirDiff.o `pkg-config --cflags --libs opencv`
g++ -I cpp/include cpp/build/dirDiff.o cpp/build/imgClass.o -o cpp/bin/dirDiff.out `pkg-config --cflags --libs opencv`

./cpp/bin/dirDiff.out  output/588017702948962343/run0000/  targets/588017702948962343/588017702948962343.calibrated.png  targets/588017702948962343/588017702948962343.info.txt
