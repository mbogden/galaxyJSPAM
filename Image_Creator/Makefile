# Make file for Image creation cpp
#

# Settings
C 			:= g++
CFLAGS 		:= `pkg-config opencv --cflags --libs`
CSRC 		:= cpp/src
CINCLUDE	:= cpp/include
CBIN 		:= cpp/bin
CBUILD 		:= cpp/build

MKDIR 		:= mkdir -p

all: setup img

setup:
	@$(MKDIR) cpp/build
	@$(MKDIR) $(CBIN)

img:
	g++ `pkg-config opencv --cflags --libs` -I cpp/include -c cpp/src/imgClass.cpp -o cpp/build/imgClass.o
	g++ `pkg-config opencv --cflags --libs` -I cpp/include -c cpp/src/imgCreator.cpp -o cpp/build/imgCreator.o
	g++ `pkg-config opencv --cflags --libs` -I cpp/include -c cpp/src/img.cpp -o cpp/build/img.o
	g++ cpp/build/imgClass.o cpp/build/imgCreator.o cpp/build/img.o -o cpp/bin/img.exe `pkg-config opencv --cflags --libs` -fopenmp  -I cpp/include 


