#ifndef CNN_h
#define CNN_h

#include <cmath>
#include <array>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <limits>

#define IMAGE_DIM 28

using std::floor;
using std::exp;
using std::log;
using std::normal_distribution;
using std::sqrt;
using std::default_random_engine;
using std::string;
using std::vector;
using std::end;
using std::begin;
using std::for_each;

template<typename T>
using array2D = vector<vector<T> >;

template<typename T>
using array3D = vector<vector<vector<T> > >;

template<typename T>
using array4D = vector<vector<vector<vector<T> > > >;

class Conv_Layer {
public:
    Conv_Layer(int num_filters, int filter_dim, int stride);
    array3D<double> forward(array3D<double> &image);
    array3D<double> backward(array3D<double> &dprev);
    int get_out_dim();
private:
    array3D<double> filters;
    vector<double> bias;
    int num_filters;
    int stride;
    int filter_dim;
    int out_dim;
    normal_distribution normal_dist;
    array3D<double> image; // save this for back prop
    array3D<double> df;
    vector<double> dB;
};

class Dense_Layer {
public:
    Dense_Layer(int in_dim, int out_dim);
    vector<double> forward(vector<double> &in);
    vector<double> backward(vector<double> &dprev);
private:
    array2D<double> weights;
    array2D<double> weights_T; // update this after each mini batch, after adam is run
    vector<double> bias;
    array2D<double> dW;
    vector<double> dB;
    vector<double> orig_in; // save this for back prop
    int in_dim;
    int out_dim;
    normal_distribution normal_dist;
};

class MaxPool_Layer {
public:
    MaxPool_Layer(int kernel_size, int stride);
    array3D<double> forward(array3D<double> &processed_image);
    array3D<double> backward(array3D<double> &dprev);
private:
    void argmax(int channel_num, int curr_y, int curr_x, int &y_max, int &x_max);
    int kernel_size;
    int stride;
    array3D<double> orig_image; // save this for back prop
};

class Model {
public:
    Model(int filter_dim, int pool_dim, int num_filters, int pool_stride, int conv_stride);
    void backprop(vector<double> &probs, vector<int> &labels_one_hot);
private:
    vector<Dense_Layer> dense_layers;
    vector<Conv_Layer> conv_layers;
    MaxPool_Layer maxpool_layer;
    int filter_dim;
    int pool_dim;
    int num_filters;
    int pool_stride;
    int conv_stride;
};

array3D<double> unflatten(vector<double> &vec, int num_filters, int pool_dim);
vector<double> flatten(array3D<double> &image);
void relu(vector<double> &in);
void relu(array3D<double> &in);
void softmax(vector<double> &in);
double cat_cross_entropy(vector<double> &pred_probs, vector<double> &true_labels);
array2D<double> rotate_180(array2D<double> &filter);
array2D<double> transpose(array2D<double> &w);
vector<double> dot_product(array2D<double> &w, vector<double> &x);

#endif