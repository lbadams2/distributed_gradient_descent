// Client side C/C++ program to demonstrate Socket programming
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "cnn.h"
#include <iostream>

using std::cout;
using std::endl;

#define PORT 8080

array4D<float> get_images(int num_images) {
    int image_dim = 5;
    vector<vector<vector<vector<float> > > > images(num_images, vector<vector<vector<float> > >(1, vector<vector<float> >(image_dim, vector<float>(image_dim))));
    normal_distribution<float> normal_dist = normal_distribution<float>(0, 1);
    default_random_engine generator;
    for(int n = 0; n < num_images; n++)
        for(int i = 0; i < image_dim; i++)
            for(int j = 0; j < image_dim; j++)
                images[n][0][i][j] = normal_dist(generator);
    return images;
}

vector<float> get_bias(int out_dim) {
    vector<float> bias(out_dim);
    normal_distribution<float> normal_dist = normal_distribution<float>(0, 1);
    default_random_engine generator;
    for(int i = 0; i < out_dim; i++)
        bias[i] = normal_dist(generator);
    return bias;
}

array2D<float> get_dense_weights(int out_dim, int in_dim) {
    vector<vector<float>> weights(out_dim, vector<float>(in_dim));
    normal_distribution<float> normal_dist = normal_distribution<float>(0, 1);
    default_random_engine generator;
    for(int i = 0; i < out_dim; i++)
        for(int j = 0; j < in_dim; j++)
            weights[i][j] = normal_dist(generator);
    return weights;
}

array4D<float> get_filter_weights(int num_filters, int num_channels, int filter_dim) {
    vector<vector<vector<vector<float> > > > filters(num_filters, vector<vector<vector<float> > >(num_channels, vector<vector<float> >(filter_dim, vector<float>(filter_dim))));
    normal_distribution<float> normal_dist = normal_distribution<float>(0, 1);
    default_random_engine generator;

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
   
int main(int argc, char const *argv[])
{
    int sock = 0, valread;
    struct sockaddr_in serv_addr;
    
    cout << "starting image data" << endl;
    array4D<float> images = get_images(5);
    print_images(images);
    vector<float> flattened_images = flatten_images(images);
    cout << "flattened image size " << flattened_images.size() << endl;
    //print_vec(flattened_filters);
    //float* filter_arr = flattened_filters.data();
    //print_arr(filter_arr, flattened_filters.size());
    cout << "ending image data\n\n\n";

    cout << "starting filter data" << endl;
    array4D<float> filters = get_filter_weights(4, 3, 2);
    print_filters(filters);
    vector<float> flattened_filters = flatten_filters(filters);
    cout << "flattened filter size " << flattened_filters.size() << endl;
    //print_vec(flattened_filters);
    //float* filter_arr = flattened_filters.data();
    //print_arr(filter_arr, flattened_filters.size());
    cout << "ending filter data\n\n\n";

    cout << "starting weight data" << endl;
    array2D<float> weights = get_dense_weights(3, 5);
    print_weights(weights);
    vector<float> flattened_weights = flatten_weights(weights);
    cout << "flattened weight size " << flattened_weights.size() << endl;
    //print_vec(flattened_filters);
    //float* filter_arr = flattened_filters.data();
    //print_arr(filter_arr, flattened_filters.size());
    cout << "ending weight data\n\n\n";

    cout << "starting bias data" << endl;
    vector<float> bias = get_bias(5);
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
    float* all_vec_arr = all_vec.data();
    
    char buffer[1024] = {0};

    
   
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
       
    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr)<=0) 
    {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }
    
    for(int i = 0; i < 2; i++) {   
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
        send(sock , all_vec_arr , all_vec.size() * 4 , 0 ); // 3rd arg is length in bytes, float is 4 bytes
        printf("Hello message sent\n");
        valread = read( sock , buffer, 1024);
        printf("%s\n",buffer );
    }
    return 0;
}