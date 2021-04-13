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
    vector<double> bias(out_dim, 0);
}

vector<double> Dense_Layer::forward(vector<double> &in)
{
    int rows = weights.size();
    int cols = weights[0].size();
    vector<double> product(rows);
    int tmp = 0;
    for (int i = 0; i < rows; i++)
    {
        double out_value = 0;
        for (int j = 0; j < cols; j++)
        {
            tmp = weights[i][j] * in[j];
            out_value += tmp;
        }
        product[i] = out_value;
    }

    //relu(product);
    return product;
}

// matrix matrix dot product is matrix
// matrix vector dot product is vector
// vector vector dot product is scalar
// forward operation produces a vector
// moving in backward direction, dprev is previous gradient
void Dense_Layer::backward(vector<double> &dprev, vector<double> &orig_in, vector<double> &dW, vector<double> &dB, vector<double> &d_orig_in)
{
    // dW is outer product of dprev and orig_in to get matrix
    // dB is sum of dprev along cols
    // d_orig_in is weights^T * dprev
}