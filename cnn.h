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
#define FILTER_DIM 5
#define POOL_DIM 2

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

class Conv_Layer {
public:
    Conv_Layer(int num_filters);
    array3D<double> forward(array2D<int> image, int stride);
    void backward(array3D<double> dprev, array2D<int> image, int stride, array3D<double> &df, array3D<double> &dx);
private:
    array3D<double> filters;
    vector<double> bias;
    int num_filters;
    normal_distribution normal_dist;
};

class Dense_Layer {
public:
    Dense_Layer(int in_dim, int out_dim);
    vector<double> forward(vector<double> &in);
    void backward(vector<double> &dprev, vector<double> &orig_in, vector<double> &dW, vector<double> &dB, vector<double> &d_orig_in);
private:
    array2D<double> weights;
    vector<double> bias;
    int in_dim;
    int out_dim;
    normal_distribution normal_dist;
};

class MaxPool_Layer {
public:
    MaxPool_Layer(int kernel_size, int stride);
    array2D<double> forward(array2D<double> &processed_image);
    array2D<double> backward(array2D<double> &dprev);
private:
    void argmax(int curr_y, int curr_x, int &y_max, int &x_max);
    int kernel_size;
    int stride;
    array2D<double> orig_image;
};

vector<int> flatten(array2D<int> &image);
void relu(vector<double> &in);
void softmax(vector<double> &in);
double cat_cross_entropy(vector<double> &pred_probs, vector<double> &true_labels);
array2D<int> rotate_180(array2D<int> filter);
array2D<double> transpose(array2D<double> w);

#endif