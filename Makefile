grad_calc: conv_layer.o dense_layer.o maxpool_layer.o cnn.o grad_calc.o
	g++ -std=c++17 -g -o grad_calc conv_layer.o dense_layer.o maxpool_layer.o cnn.o grad_calc.o

optimizer: load_image.o conv_layer.o dense_layer.o maxpool_layer.o cnn.o optimizer.o
	g++ -std=c++17 -g -o optimizer load_image.o conv_layer.o dense_layer.o maxpool_layer.o cnn.o optimizer.o

grad_calc.o: distributed.h cnn.h grad_calc.cc
	g++ -std=c++17 -g -c grad_calc.cc

optimizer.o: distributed.h cnn.h load_image.h optimizer.cc
	g++ -std=c++17 -g -c optimizer.cc

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

#test
server:
	g++ -std=c++17 -g -o worker worker.cc

client:
	g++ -std=c++17 -g -o master master.cc

unit_tests: unit_tests.cc
	g++ -std=c++17 -g -o unit_tests unit_tests.cc

clean:
	rm *.o train