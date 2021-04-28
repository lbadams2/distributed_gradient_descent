FROM ubuntu:20.04
RUN mkdir -p /home/dockerc
WORKDIR /home/dockerc
COPY . /home/dockerc
RUN apt -y update
RUN apt install -y g++

RUN g++ -std=c++17 -g -c load_image.cc
RUN g++ -std=c++17 -g -c conv_layer.cc
RUN g++ -std=c++17 -g -c dense_layer.cc
RUN g++ -std=c++17 -g -c maxpool_layer.cc
RUN g++ -std=c++17 -g -c cnn.cc
RUN g++ -std=c++17 -pthread -g -c optimizer.cc
RUN g++ -std=c++17 -pthread -g -o optimizer load_image.o conv_layer.o dense_layer.o maxpool_layer.o cnn.o optimizer.o

#RUN g++ -std=c++17 -pthread -g -o optimizer master.cc

RUN chmod +x /home/dockerc/optimizer
#CMD ["sh", "-c", "./worker ${port} ${seed}"]
ENTRYPOINT ["/home/dockerc/optimizer"]