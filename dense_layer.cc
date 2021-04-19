#include "cnn.h"

Dense_Layer::Dense_Layer(int in_dim, int out_dim) : in_dim(in_dim), out_dim(out_dim)
{
    // init weights and bias
    // 5 x 4 dot 4 x 1 = 5 x 1 so its out_dim x in_dim
    vector<vector<float>> weights(out_dim, vector<float>(in_dim));
    normal_dist = normal_distribution<float> distribution(0, 1);
    default_random_engine generator;
    for (int i = 0; i < in_dim; i++)
    {
        for (int j = 0; j < out_dim; j++)
        {
            float init_val = normal_dist(generator);
            weights[i][j] = init_val * .01;
        }
    }
    weights_T = transpose(weights);
    vector<float> bias(out_dim, 0);
}

vector<float> Dense_Layer::forward(vector<float> &in)
{
    vector<float> f = dot_product(weights, in);
    for(int i = 0; i < f.size(); i++)
        f[i] += bias[i];
    //relu(product);
    orig_in = f; // save this for back propagation
    return f;
}

// forward operation produces a vector
// moving in backward direction, dprev is previous gradient
// vectors are row vectors
vector<float> Dense_Layer::backward(vector<float> &dprev)
{
    // dW should have same dims as weights
    vector<vector<float> > dW(out_dim, vector<float>(in_dim, 0));
    
    // dW is outer product of dprev and orig_in to get matrix
    for(int i  = 0; i < dprev.size(); i++) {
        for(int j = 0; j < orig_in.size(); j++) {
            dW[i][j] = dprev[i] * orig_in[j];
        }
    }

    // dB is sum of dprev along cols, which i think is just dprev, dL/dprev * dprev/dB, prev = wx + b so dprev/dB = 1
    vector<float> dB(out_dim, 0);
    for(int i = 0; i < dprev.size(); i++)
        dB[i] = dprev[i];

    // d_orig_in is weights^T * dprev
    vector<float> d_orig_in = dot_product(weights_T, dprev);
    return d_orig_in;
}

int Dense_Layer::get_in_dim() {
    return in_dim;
}

array2D<float> Dense_Layer::get_dW() {
    return dW;
}

vector<float> Dense_Layer::get_dB() {
    return dB;
}