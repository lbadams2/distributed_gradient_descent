FROM ubuntu:20.04
RUN mkdir -p /home/dockerc
WORKDIR /home/dockerc
COPY . /home/dockerc
RUN apt -y update
RUN apt install -y g++
RUN g++ -std=c++17 -pthread -g -o optimizer master.cc
RUN chmod +x /home/dockerc/optimizer
#CMD ["sh", "-c", "./worker ${port} ${seed}"]
ENTRYPOINT ["/home/dockerc/optimizer"]