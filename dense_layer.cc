#include "cnn.h"

Dense_Layer::Dense_Layer(int in_dim, int out_dim) : in_dim(in_dim), out_dim(out_dim)
{
    // init weights and bias
    vector<vector<double>> weights(in_dim, vector<double>(out_dim));
    normal_dist = normal_distribution<double> distribution(0, 1);
    default_random_engine generator;
    for (int i = 0; i < in_dim; i++)
    {
        for (int j = 0; j < out_dim; j++)
        {
            double init_val = normal_dist(generator);
            weights[i][j] = init_val * .01;
        }
    }
    weights_T = transpose(weights);
    vector<double> bias(out_dim, 0);
}

vector<double> Dense_Layer::forward(vector<double> &in)
{
    vector<double> f = dot_product(weights, in);
    for(int i = 0; i < f.size(); i++)
        f[i] += bias[i];
    //relu(product);
    orig_in = f; // save this for back propagation
    return f;
}

// forward operation produces a vector
// moving in backward direction, dprev is previous gradient
// vectors are row vectors
vector<double> Dense_Layer::backward(vector<double> &dprev)
{
    // dW is outer product of dprev and orig_in to get matrix
    for(int i  = 0; i < dprev.size(); i++) {
        for(int j = 0; j < orig_in.size(); j++) {
            dW[i][j] = dprev[i] * orig_in[j];
        }
    }

    // dB is sum of dprev along cols, which i think is just dprev, dL/dprev * dprev/dB, prev = wx + b so dprev/dB = 1
    for(int i = 0; i < dprev.size(); i++)
        dB[i] = dprev[i];

    // d_orig_in is weights^T * dprev
    vector<double> d_orig_in = dot_product(weights_T, dprev);
    return d_orig_in;
}