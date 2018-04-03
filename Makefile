#Standard C++
CC=g++-7 -O3 -Wno-deprecated
STD=-std=c++11
CFLAGS=-framework OpenGL -framework GLUT -fopenmp 
LIB= -lm -ldl -lrt
LDFLAGS=  -L/usr/local/opt/llvm/lib
CPPFLAGS= -I/usr/local/opt/llvm/include

#Cuda
CU=nvcc -O3 -w
CUSTD=#-Xcompiler -std=c++
CUFLAGS=-Xlinker -framework,OpenGL,-framework,GLUT


all: cudaSim.o
	$(CU) $(CUSTD) $(CUFLAGS) cudaSim.o -o cmain

cudaSim.o: cudaSim.cu
	$(CU) $(CUSTD) $(CUFLAGS) cudaSim.cu -c

nvp:
	$(CU) $(CUSTD) $(CUFLAGS) cudaSim.o -o cmain
	nvprof ./cmain

clean:
	rm -rf *o cmain *.gch
