all: train

train: load_image.o conv_layer.o dense_layer.o maxpool_layer.o cnn.o train.o
	g++ -std=c++17 -g -o train load_image.o conv_layer.o dense_layer.o maxpool_layer.o cnn.o train.o

train.o: load_image.h cnn.h train.cc
	g++ -std=c++17 -g -c train.cc

cnn.o: cnn.h cnn.cc
	g++ -std=c++17 -g -c cnn.cc

maxpool_layer.o: cnn.h maxpool_layer.cc
	g++ -std=c++17 -g -c maxpool_layer.cc

dense_layer.o: cnn.h dense_layer.cc
	g++ -std=c++17 -g -c dense_layer.cc

conv_layer.o: cnn.h conv_layer.cc
	g++ -std=c++17 -g -c conv_layer.cc

load_image.o: load_image.h load_image.cc
	g++ -std=c++17 -g -c load_image.cc

clean:
	rm *.o train