#include "cnn.h"

Dense_Layer::Dense_Layer(int in_dim, int out_dim) : in_dim(in_dim), out_dim(out_dim)
{
    // init weights and bias
    // 5 x 4 dot 4 x 1 = 5 x 1 so its out_dim x in_dim
    vector<vector<float>> weights(out_dim, vector<float>(in_dim));
    vector<float> flattened_weights(out_dim * in_dim);

    normal_distribution<float> normal_dist = normal_distribution<float>(0, 1);
    default_random_engine generator;
    int vec_idx = 0;
    for (int i = 0; i < out_dim; i++)
    {
        for (int j = 0; j < in_dim; j++)
        {
            float init_val = normal_dist(generator);
            weights[i][j] = init_val * .01;
            flattened_weights[vec_idx++] = init_val * .01;
        }
    }
    this->weights = weights;
    this->flattened_weights = flattened_weights;
    this->weights_T = transpose(weights);
    vector<float> bias(out_dim, 0);
    this->bias = bias;

    vector<vector<float> > dW(out_dim, vector<float>(in_dim, 0));
    this->dW = dW;

    vector<float> dW_flattened(out_dim * in_dim, 0);
    this->dW_flattened = dW_flattened;

    vector<float> dB(out_dim, 0);
    this->dB = dB;
}

Dense_Layer::Dense_Layer(){}

vector<float> Dense_Layer::forward(vector<float> &in)
{
    vector<float> f = dot_product(weights, in);
    for(int i = 0; i < f.size(); i++)
        f[i] += bias[i];
    //relu(product);
    //orig_in = f; // save this for back propagation
    this->orig_in = in;
    return f;
}

// forward operation produces a vector
// moving in backward direction, dprev is previous gradient
// vectors are row vectors
vector<float> Dense_Layer::backward(vector<float> &dprev, bool reset_grads)
{
    /*
    if(reset_grads) {
        // dW should have same dims as weights
        vector<vector<float> > dW(out_dim, vector<float>(in_dim, 0));
        this->dW = dW;
        vector<float> dB(out_dim, 0);
        this->dB = dB;
    }
    */

    if(reset_grads)
        this->weights_T = transpose(weights);
    
    int dw_index = 0;
    // dW is outer product of dprev and orig_in to get matrix
    for(int i  = 0; i < dprev.size(); i++) {
        for(int j = 0; j < orig_in.size(); j++) {
            if(reset_grads) {
                dW[i][j] = dprev[i] * orig_in[j];
                dW_flattened[dw_index++] = dprev[i] * orig_in[j];
            }
            else {
                dW[i][j] += dprev[i] * orig_in[j];
                dW_flattened[dw_index++] += dprev[i] * orig_in[j];
            }
        }
    }

    // dB is sum of dprev along cols, which i think is just dprev, dL/dprev * dprev/dB, prev = wx + b so dprev/dB = 1
    for(int i = 0; i < dprev.size(); i++)
        if(reset_grads)
            dB[i] = dprev[i];
        else
            dB[i] += dprev[i];

    // d_orig_in is weights^T * dprev, this isn't summed over the batch
    vector<float> d_orig_in = dot_product(weights_T, dprev);
    return d_orig_in;
}

vector<float>& Dense_Layer::get_flattened_weights() {
    return flattened_weights;
}

vector<float>& Dense_Layer::get_flattened_dW() {
    return dW_flattened;
}

/*
vector<float>& Dense_Layer::get_flattened_weights() {
    vector<float> flattened(out_dim * in_dim);
    int vec_idx = 0;
    for(int i = 0; i < out_dim; i++)
        for(int j = 0; j < in_dim; j++)
            flattened[vec_idx++] = weights[i][j];
    return flattened;
}
*/

int Dense_Layer::get_in_dim() {
    return in_dim;
}

array2D<float>& Dense_Layer::get_dW() {
    return dW;
}

vector<float>& Dense_Layer::get_dB() {
    return dB;
}

void Dense_Layer::set_dW(array2D<float> &dW) {
    this->dW = dW;
}

void Dense_Layer::set_dB(vector<float> &dB) {
    this->dB = dB;
}

array2D<float>& Dense_Layer::get_weights() {
    return weights;
}

vector<float>& Dense_Layer::get_bias() {
    return bias;
}
