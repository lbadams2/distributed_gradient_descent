To build the grad_calc docker image run `docker build -t grad_calc:1 -f grad_calc.dockerfile .`

To build the optimizer docker image run `docker build -t optimizer:1 -f optimizer.dockerfile .`

To run the distributed system `docker-compose up -d`

To get logs run `docker-compose logs -t -f grad_calc_1` or `docker-compose logs -t -f optimizer`

The optimizer sleeps while waiting for the grad_calc responses, so to get the runtime you have to add the optimizer
time to the longest running grad_calc time

To compile the single node run `make single_node_train`. To run `single_node_train 32` with 32 the batch size, can be any number

Data was downloaded from http://yann.lecun.com/exdb/mnist/

To compile with opencv
clang++ $(pkg-config --cflags --libs /usr/local/Cellar/opencv/4.5.2/lib/pkgconfig/opencv4.pc) -std=c++17  unit_tests.cc -o unit_tests

https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c

https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1

