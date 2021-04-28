FROM ubuntu:20.04
RUN mkdir -p /home/dockerc
WORKDIR /home/dockerc
COPY . /home/dockerc
RUN apt -y update
RUN apt install -y g++

RUN g++ -std=c++17 -g -c cnn.cc
RUN g++ -std=c++17 -g -c maxpool_layer.cc
RUN g++ -std=c++17 -g -c dense_layer.cc
RUN g++ -std=c++17 -g -c conv_layer.cc
RUN g++ -std=c++17 -g -c grad_calc.cc
RUN g++ -std=c++17 -g -o grad_calc conv_layer.o dense_layer.o maxpool_layer.o cnn.o grad_calc.o

#RUN g++ -std=c++17 -g -o worker worker.cc

RUN chmod +x /home/dockerc/grad_calc
#EXPOSE 8080
#CMD ["sh", "-c", "./worker ${port} ${seed}"]
ENTRYPOINT ["/home/dockerc/grad_calc"]