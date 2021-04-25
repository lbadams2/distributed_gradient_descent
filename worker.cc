// Server side C/C++ program to demonstrate Socket programming
#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <iostream>
#include "cnn.h"

using std::cout;
using std::endl;

#define PORT 8080
#define NUM_IMAGES 5
#define NUM_FILTERS 4
#define FILTER_DIM 2
#define FILTER_CHANNELS 3
#define DENSE_FIRST_OUT 3
#define DENSE_FIRST_IN 5
#define TEST_IMAGE_DIM 5
#define BIAS_DIM 5

void print_arr(float* filter_arr, int size) {
    cout << "printing arr, size " << size << endl;    
    for(int i = 0; i < size; i++) {
        cout << filter_arr[i] << " ";
    }
    cout << endl;
}

void read_buf(float* buf, array4D<float> &images, array4D<float> &filters, array2D<float> &weights, vector<float> &bias) {
    int image_start = 0;
    int image_end = NUM_IMAGES * TEST_IMAGE_DIM * TEST_IMAGE_DIM;
    int filter_start = image_end;
    int filter_end = filter_start + (NUM_FILTERS * FILTER_CHANNELS * FILTER_DIM * FILTER_DIM);
    int weight_start = filter_end;
    int weight_end = weight_start + (DENSE_FIRST_IN * DENSE_FIRST_OUT);
    int bias_start = weight_end;
    int bias_end = bias_start + BIAS_DIM;

    cout << "printing image" << endl;
    int image_idx = image_start;
    for(int n = 0; n < NUM_IMAGES; n++)
        for(int i = 0; i < TEST_IMAGE_DIM; i++)
            for(int j = 0; j < TEST_IMAGE_DIM; j++) {
                images[n][0][i][j] = buf[image_idx++];
                cout << images[n][0][i][j] << endl;
            }
    cout << "image idx: " << image_idx << endl;
    cout << "image end: " << image_end << endl;
    cout << "filter start: " << filter_start << endl;
    assert(image_idx == image_end);
    cout << "done printing image\n\n\n";

    cout << "printing filter" << endl;
    int filter_idx = filter_start;
    for(int f = 0; f < NUM_FILTERS; f++)
        for(int n = 0; n < FILTER_CHANNELS; n++)
            for(int i = 0; i < FILTER_DIM; i++)
                for(int j = 0; j < FILTER_DIM; j++) {
                    filters[f][n][i][j] = buf[filter_idx++];
                    cout << filters[f][n][i][j] << endl;
                }
    assert(filter_idx == filter_end);
    cout << "done printing filter\n\n\n";

    cout << "printing weights" << endl;
    int weight_idx = weight_start;
    for(int i = 0; i < DENSE_FIRST_OUT; i++)
        for(int j = 0; j < DENSE_FIRST_IN; j++) {
            weights[i][j] = buf[weight_idx++];
            cout << weights[i][j] << endl;
        }
    assert(weight_idx == weight_end);
    cout << "done printing weights\n\n\n";

    cout << "printing bias" << endl;
    int bias_idx = bias_start;
    for(int i = 0; i < BIAS_DIM; i++) {
        bias[i] = buf[bias_idx++];
        cout << bias[i] << endl;
    }
    assert(bias_idx == bias_end);
    cout << "done printing bias\n\n\n";
}

int main(int argc, char const *argv[])
{
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
        
    char *hello = "Hello from server";
       
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
       
    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( PORT );
       
    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address, 
                                 sizeof(address))<0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    int buf_len = NUM_IMAGES * TEST_IMAGE_DIM * TEST_IMAGE_DIM * DENSE_FIRST_IN * DENSE_FIRST_OUT * NUM_FILTERS * FILTER_CHANNELS * FILTER_DIM * FILTER_DIM * BIAS_DIM;        
    float buffer[NUM_IMAGES * TEST_IMAGE_DIM * TEST_IMAGE_DIM * DENSE_FIRST_IN * DENSE_FIRST_OUT * NUM_FILTERS * FILTER_CHANNELS * FILTER_DIM * FILTER_DIM * BIAS_DIM] = {0};
    int loop_idx = 0;
    while(true) {        
        cout << "reading socket loop_idx: " << loop_idx << endl;
        if (listen(server_fd, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }    

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, 
                       (socklen_t*)&addrlen))<0)
        {
            perror("accept");
            exit(EXIT_FAILURE);
        }
        valread = read( new_socket , buffer, buf_len); // buf_len in bytes
        //printf("%f\n",buffer );
        //print_arr(buffer, 48);
        vector<vector<vector<vector<float> > > > images(NUM_IMAGES, vector<vector<vector<float> > >(1, vector<vector<float> >(TEST_IMAGE_DIM, vector<float>(TEST_IMAGE_DIM))));
        vector<vector<vector<vector<float> > > > filters(NUM_FILTERS, vector<vector<vector<float> > >(FILTER_CHANNELS, vector<vector<float> >(FILTER_DIM, vector<float>(FILTER_DIM))));
        vector<vector<float>> weights(DENSE_FIRST_OUT, vector<float>(DENSE_FIRST_IN));
        vector<float> bias(BIAS_DIM);
        read_buf(buffer, images, filters, weights, bias);

        send(new_socket , hello , strlen(hello) , 0 );
        printf("Hello message sent\n");
        loop_idx++;
    }
    return 0;
}