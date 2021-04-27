// Client side C/C++ program to demonstrate Socket programming
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <netdb.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include "cnn.h"
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

using std::cout;
using std::endl;
using std::thread;
using std::mutex;
using std::lock_guard;
using std::string;

#define NUM_IMAGES 5
#define NUM_FILTERS 4
#define FILTER_DIM 2
#define FILTER_CHANNELS 3
#define DENSE_FIRST_OUT 3
#define DENSE_FIRST_IN 5
#define TEST_IMAGE_DIM 5
#define BIAS_DIM 5

mutex grad_mutex;
mutex cout_mutex;
array2D<float> dW;
vector<float> dB;

array4D<float> get_images(int num_images, normal_distribution<float> &normal_dist, default_random_engine &generator) {
    int image_dim = 5;
    vector<vector<vector<vector<float> > > > images(num_images, vector<vector<vector<float> > >(1, vector<vector<float> >(image_dim, vector<float>(image_dim))));
    for(int n = 0; n < num_images; n++)
        for(int i = 0; i < image_dim; i++)
            for(int j = 0; j < image_dim; j++)
                images[n][0][i][j] = normal_dist(generator);
    return images;
}

vector<float> get_bias(int out_dim, normal_distribution<float> &normal_dist, default_random_engine &generator) {
    vector<float> bias(out_dim);
    for(int i = 0; i < out_dim; i++)
        bias[i] = normal_dist(generator);
    return bias;
}

array2D<float> get_dense_weights(int out_dim, int in_dim, normal_distribution<float> &normal_dist, default_random_engine &generator) {
    vector<vector<float>> weights(out_dim, vector<float>(in_dim));
    for(int i = 0; i < out_dim; i++)
        for(int j = 0; j < in_dim; j++)
            weights[i][j] = normal_dist(generator);
    return weights;
}

array4D<float> get_filter_weights(int num_filters, int num_channels, int filter_dim, normal_distribution<float> &normal_dist, default_random_engine &generator) {
    vector<vector<vector<vector<float> > > > filters(num_filters, vector<vector<vector<float> > >(num_channels, vector<vector<float> >(filter_dim, vector<float>(filter_dim))));

    for(int f = 0; f < num_filters; f++)
        for(int n = 0; n < num_channels; n++)
            for(int i = 0; i < filter_dim; i++)
                for(int j = 0; j < filter_dim; j++) {
                    float val = normal_dist(generator);
                    filters[f][n][i][j] = val;
                }
    return filters;
}

vector<float> flatten_weights(array2D<float> &weights) {
    int out_dim = weights.size();
    int in_dim = weights[0].size();
    vector<float> flattened(out_dim * in_dim);
    int vec_idx = 0;
    for(int i = 0; i < out_dim; i++)
        for(int j = 0; j < in_dim; j++)
            flattened[vec_idx++] = weights[i][j];
    return flattened;
}

vector<float> flatten_images(array4D<float> &images) {
    int num_images = images.size();
    int num_channels = images[0].size();
    int image_dim = images[0][0].size();
    vector<float> flattened(num_images * num_channels * image_dim * image_dim);
    int vec_idx = 0;
    for(int n = 0; n < num_images; n++)
        for(int i = 0; i < image_dim; i++)
            for(int j = 0; j < image_dim; j++)
                flattened[vec_idx++] = images[n][0][i][j];
    return flattened;
}

vector<float> flatten_filters(array4D<float> &filters) {
    int num_filters = filters.size();
    int num_channels = filters[0].size();
    int filter_dim = filters[0][0].size();
    int flattened_dim = num_filters * num_channels * filter_dim * filter_dim;
    vector<float> flattened(flattened_dim);
    int vec_idx = 0;
    for (int f = 0; f < num_filters; f++)
    {
        for(int n = 0; n < num_channels; n++) {
            //int channel_offset = (f * num_channels) + n;
            for (int i = 0; i < filter_dim; i++)
            {
                for (int j = 0; j < filter_dim; j++)
                {
                    //int pixel_offset = (i * filter_dim) + j; // offset within filter
                    //int offset = channel_offset + pixel_offset;
                    flattened[vec_idx++] = filters[f][n][i][j];
                }
            }
        }
    }
    return flattened;
}

void print_arr(float* filter_arr, int size) {
    cout << "printing arr, size " << size << endl;    
    for(int i = 0; i < size; i++) {
        cout << filter_arr[i] << " ";
    }
    cout << endl;
    cout << endl;
}

void print_vec(vector<float> &vec) {
    cout << "printing vec" << endl;
    for(float val : vec)
        cout << val << " ";
    cout << endl;
    cout << endl;
}

void print_filters(array4D<float> &filters) {
    cout << "printing filters" << endl;
    int num_filters = filters.size();
    int num_channels = filters[0].size();
    int filter_dim = filters[0][0].size();
    for(int f = 0; f < num_filters; f++)
        for(int n = 0; n < num_channels; n++)
            for(int i = 0; i < filter_dim; i++)
                for(int j = 0; j < filter_dim; j++)
                    cout << filters[f][n][i][j] << " " << endl;
    cout << endl;
    cout << endl;
}

void print_images(array4D<float> &images) {
    cout << "printing images" << endl;
    int num_images = images.size();
    int image_dim = images[0][0].size();
    for(int n = 0; n < num_images; n++)
        for(int i = 0; i < image_dim; i++)
            for(int j = 0; j < image_dim; j++)
                cout << images[n][0][i][j] << " " << endl;
    cout << endl;
    cout << endl;
}

void print_weights(array2D<float> &images) {
    cout << "printing weights" << endl;
    int out_dim = images.size();
    int in_dim = images[0].size();
    for(int i = 0; i < out_dim; i++)
        for(int j = 0; j < in_dim; j++)
            cout << images[i][j] << " " << endl;
    cout << endl;
    cout << endl;
}

void print_from_thread(string msg, int thread_id) {
    lock_guard<mutex> guard(cout_mutex);
    cout << "thread " << thread_id << ": " << msg << endl;
}

void print_buf(float* buf, int thread_id, int buf_len) {
    lock_guard<mutex> guard(cout_mutex);
    cout << "thread " << thread_id << ": " << "printing buf" << endl;
    for(int i = 0; i < buf_len; i++)
        cout << "thread " << thread_id << ": " << buf[i] << endl;
}

void add_grads(array2D<float> &thread_dW, vector<float> &thread_dB) {
    lock_guard<mutex> guard(grad_mutex);
    for(int i = 0; i < DENSE_FIRST_OUT; i++) {
        dB[i] += thread_dB[i];
        //cout << "dB " << dB[i] << endl;
        for(int j = 0; j < DENSE_FIRST_IN; j++) {
            dW[i][j] += thread_dW[i][j];
            //cout << "dW " << dW[i][j] << endl;
        }
    }
}

void read_buf(float *buf, array2D<float> &thread_dW, vector<float> &thread_dB) {
    int weight_start = 0;
    int weight_end = DENSE_FIRST_OUT * DENSE_FIRST_IN;
    int bias_start = weight_end;
    int weight_idx = weight_start;
    for(int i = 0; i < DENSE_FIRST_OUT; i++)
        for(int j = 0; j < DENSE_FIRST_IN; j++)
            thread_dW[i][j] = buf[weight_idx++];
    assert(weight_idx == bias_start);
    
    int bias_idx = bias_start;
    for(int i = 0; i < BIAS_DIM; i++)
        thread_dB[i] = buf[bias_idx++];
}

int send_vec(float* all_vec_arr, int vec_size, const char* ip_address, int port, int thread_id) {
    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    float buffer[19] = {0};
    int buf_len = 19 * 4;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
       
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, ip_address, &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }
    
       
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return -1;
    } 

    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
    {
            printf("\nConnection Failed \n");
            return -1;
    }            
    send(sock , all_vec_arr , vec_size * 4 , 0 ); // 3rd arg is length in bytes, float is 4 bytes
    print_from_thread("Hello message sent", thread_id);
    valread = read( sock , buffer, buf_len);
    print_buf(buffer, thread_id, buf_len / 4);

    array2D<float> thread_dW(DENSE_FIRST_OUT, vector<float>(DENSE_FIRST_IN, 0));
    vector<float> thread_dB(BIAS_DIM, 0);
    read_buf(buffer, thread_dW, thread_dB);
    add_grads(thread_dW, thread_dB);
    //printf("%s\n",buffer );
    return 0;
}

void print_grads() {
    cout << "printing global dW" << endl;
    for(int i = 0; i < DENSE_FIRST_OUT; i++)
        for(int j = 0; j < DENSE_FIRST_IN; j++)
            cout << dW[i][j] << endl;
    
    cout << "printing global dB" << endl;
    for(int i = 0; i < BIAS_DIM; i++)
        cout << dB[i] << endl;
}

vector<float> get_vec(normal_distribution<float> &normal_dist, default_random_engine &generator) {
    cout << "starting image data" << endl;
    array4D<float> images = get_images(NUM_IMAGES, normal_dist, generator);
    print_images(images);
    vector<float> flattened_images = flatten_images(images);
    cout << "flattened image size " << flattened_images.size() << endl;
    //print_vec(flattened_filters);
    //float* filter_arr = flattened_filters.data();
    //print_arr(filter_arr, flattened_filters.size());
    cout << "ending image data\n\n\n";

    cout << "starting filter data" << endl;
    array4D<float> filters = get_filter_weights(NUM_FILTERS, FILTER_CHANNELS, FILTER_DIM, normal_dist, generator);
    print_filters(filters);
    vector<float> flattened_filters = flatten_filters(filters);
    cout << "flattened filter size " << flattened_filters.size() << endl;
    //print_vec(flattened_filters);
    //float* filter_arr = flattened_filters.data();
    //print_arr(filter_arr, flattened_filters.size());
    cout << "ending filter data\n\n\n";

    cout << "starting weight data" << endl;
    array2D<float> weights = get_dense_weights(DENSE_FIRST_IN, DENSE_FIRST_OUT, normal_dist, generator);
    print_weights(weights);
    vector<float> flattened_weights = flatten_weights(weights);
    cout << "flattened weight size " << flattened_weights.size() << endl;
    //print_vec(flattened_filters);
    //float* filter_arr = flattened_filters.data();
    //print_arr(filter_arr, flattened_filters.size());
    cout << "ending weight data\n\n\n";

    cout << "starting bias data" << endl;
    vector<float> bias = get_bias(BIAS_DIM, normal_dist, generator);
    print_vec(bias);
    cout << "flattened bias size " << bias.size() << endl;
    //print_vec(flattened_filters);
    //float* filter_arr = flattened_filters.data();
    //print_arr(filter_arr, flattened_filters.size());
    cout << "ending bias data\n\n\n";

    vector<float> all_vec;
    all_vec.reserve(flattened_images.size() * flattened_filters.size() * flattened_weights.size() * bias.size());
    all_vec.insert( all_vec.end(), flattened_images.begin(), flattened_images.end() );
    all_vec.insert( all_vec.end(), flattened_filters.begin(), flattened_filters.end() );
    all_vec.insert( all_vec.end(), flattened_weights.begin(), flattened_weights.end() );
    all_vec.insert( all_vec.end(), bias.begin(), bias.end() );
    
    return all_vec;
}

string resolve_host(const char* host_name) {
    struct hostent *host_entry;
    const char *IPbuffer;

    host_entry = gethostbyname(host_name);
    IPbuffer = inet_ntoa(*((struct in_addr*)host_entry->h_addr_list[0]));    
    return IPbuffer;
}
   
int main(int argc, char const *argv[])
{       
    normal_distribution<float> normal_dist = normal_distribution<float>(0, 1);
    default_random_engine generator;

    cout << "sleeping for 5 seconds to let workers start" << endl;
    std::chrono::milliseconds timespan(5000);
    std::this_thread::sleep_for(timespan);

    const char* hostname_1 = "grad_calc_1";
    const char* hostname_2 = "grad_calc_2";

    string ip_1_str = resolve_host(hostname_1);
    string ip_2_str = resolve_host(hostname_2);
    const char * ip_1 = ip_1_str.c_str();
    const char * ip_2 = ip_2_str.c_str();
    cout << "grad_calc_1 IP: " << ip_1 << endl; 
    cout << "grad_calc_2 IP: " << ip_2 << endl;

    for(int i = 0; i < 2; i++) {
        vector<float> all_vec_t1 = get_vec(normal_dist, generator);
        vector<float> all_vec_t2 = get_vec(normal_dist, generator);
        float* all_vec_t1_arr = all_vec_t1.data();
        float* all_vec_t2_arr = all_vec_t2.data();
        dW = array2D<float>(DENSE_FIRST_OUT, vector<float>(DENSE_FIRST_IN, 0));
        dB = vector<float>(BIAS_DIM, 0);
        thread t1(send_vec, all_vec_t1_arr, all_vec_t1.size(), ip_1, 8080, 1);
        thread t2(send_vec, all_vec_t2_arr, all_vec_t2.size(), ip_2, 8081, 2);
        t1.join();
        t2.join();
        cout << "printing grads iteration " << i << endl;
        print_grads();
    }
    
    return 0;
}