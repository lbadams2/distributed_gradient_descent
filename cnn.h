#ifndef CNN_h
#define CNN_h

#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <limits>
#include <assert.h>
#include "shared.h"

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
using std::pow;

class Conv_Layer {
public:
    Conv_Layer(int num_filters, int filter_dim, int num_channels, int stride, int image_dim);
    Conv_Layer(); // for vector init
    array3D<float> forward(array3D<float> &image);
    array3D<float> backward(array3D<float> &dprev, bool reset_grads);
    int get_out_dim();
    array4D<float>& get_dF();
    vector<float>& get_dB();
    array4D<float>& get_filters();
    vector<float>& get_bias();
    vector<float>& get_flattened_filter();
private:
    array4D<float> filters;
    vector<float> flattened_filters;
    vector<float> bias;
    int num_filters;
    int num_channels;
    int stride;
    int filter_dim;
    int image_dim;
    int out_dim;
    array3D<float> image; // save this for back prop
    array4D<float> df;
    vector<float> dB;
};

class Dense_Layer {
public:
    Dense_Layer(int in_dim, int out_dim);
    Dense_Layer(); // for vector init
    vector<float> forward(vector<float> &in);
    vector<float> backward(vector<float> &dprev, bool reset_grads);
    int get_in_dim();
    vector<float>& get_flattened_weights();
    array2D<float>& get_dW();
    vector<float>& get_dB();
    array2D<float>& get_weights();
    vector<float>& get_bias();
private:
    array2D<float> weights;
    vector<float> flattened_weights;
    array2D<float> weights_T; // update this after each mini batch, after adam is run
    vector<float> bias;
    array2D<float> dW;
    vector<float> dB;
    vector<float> orig_in; // save this for back prop
    int in_dim;
    int out_dim;
};

class MaxPool_Layer {
public:
    MaxPool_Layer(int kernel_size, int stride);
    array3D<float> forward(array3D<float> &processed_image);
    array3D<float> backward(array3D<float> &dprev);
    int get_out_dim(int in_dim);
private:
    void argmax(int channel_num, int curr_y, int curr_x, int &y_max, int &x_max);
    int kernel_size;
    int stride;
    array3D<float> orig_image; // save this for back prop
};

class Model {
public:
    Model(int filter_dim, int pool_dim, int num_filters, int pool_stride, int conv_stride, int dense_first_out_dim, float learning_rate, float beta1, float beta2, int batch_size);
    void backprop(vector<float> &probs, vector<uint8_t> &labels_one_hot, bool reset_grads);
    vector<float> forward(array3D<float> &image, vector<uint8_t> &label_one_hot);
    vector<Dense_Layer>& get_dense_layers();
    vector<Conv_Layer>& get_conv_layers();
    void adam();
private:
    vector<Dense_Layer> dense_layers;
    vector<Conv_Layer> conv_layers;
    MaxPool_Layer maxpool_layer;
    int filter_dim;
    int pool_dim;
    int dense_first_out_dim;
    int num_filters;
    int pool_stride;
    int conv_stride;
    float learning_rate;
    float beta1;
    float beta2;
    int batch_size;
    
    array4D<float> m_df1;
    vector<float> m_db1;
    array4D<float> v_df1;
    vector<float> v_db1;
    
    array4D<float> m_df2;
    vector<float> m_db2;
    array4D<float> v_df2;
    vector<float> v_db2;
    
    array2D<float> m_dw1;
    vector<float> m_db3;
    array2D<float> v_dw1;
    vector<float> v_db3;
    
    array2D<float> m_dw2;
    vector<float> m_db4;
    array2D<float> v_dw2;
    vector<float> v_db4;
};

array3D<float> unflatten(vector<float> &vec, int num_filters, int pool_dim);
vector<float> flatten(array3D<float> &image);
void relu(vector<float> &in);
void relu(array3D<float> &in);
void softmax(vector<float> &in);
float cat_cross_entropy(vector<float> &pred_probs, vector<uint8_t> &true_labels);
array2D<float> rotate_180(array2D<float> filter);
array2D<float> transpose(array2D<float> &w);
vector<float> dot_product(array2D<float> &w, vector<float> &x);

#endif
