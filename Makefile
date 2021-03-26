all: load_image

load_image: load_image.o
	g++ -std=c++17 -g -o load_image.out load_image.o

load_image.o: load_image.cc
	g++ -std=c++17 -g -c load_image.cc

clean:
	rm *.o *.out