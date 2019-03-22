# Simple 

g++ -I cpp/include -c Comparison_Methods/Direct_Comparison/src/imgClass.cpp -o Comparison_Methods/Direct_Comparison/build/imgClass.o `pkg-config --cflags --libs opencv`

g++ -I cpp/include -c Comparison_Methods/Direct_Comparison/src/diffAll.cpp -o Comparison_Methods/Direct_Comparison/build/diffAll.o `pkg-config --cflags --libs opencv`

g++ -I cpp/include Comparison_Methods/Direct_Comparison/build/diffAll.o Comparison_Methods/Direct_Comparison/build/imgClass.o -o Comparison_Methods/Direct_Comparison/bin/diffAll.out `pkg-config --cflags --libs opencv`


./Comparison_Methods/Direct_Comparison/bin/diffAll.out output/588017702948962343/ targets/588017702948962343/588017702948962343.calibrated.png targets/588017702948962343/588017702948962343.info.txt

