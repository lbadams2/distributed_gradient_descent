#include "cnn.h"

Conv_Layer::Conv_Layer(int nf, int filter_dim, int num_channels, int stride, int image_dim) : num_filters(nf), filter_dim(filter_dim), num_channels(num_channels), stride(stride), image_dim(image_dim)
{
    // filters initialized using standard normal, bias initialized all zeroes
    vector<vector<vector<vector<float> > > > filters(num_filters, vector<vector<vector<float> > >(num_channels, vector<vector<float> >(filter_dim, vector<float>(filter_dim))));
    float stddev = 1 / sqrt(filter_dim * filter_dim);
    normal_distribution<float> normal_dist = normal_distribution<float>(0, stddev);
    default_random_engine generator;
    for (int i = 0; i < num_filters; i++)
    {
        for(int n = 0; n < num_channels; n++) {
            for (int j = 0; j < filter_dim; j++)
            {
                for (int k = 0; k < filter_dim; k++)
                {
                    float init_val = normal_dist(generator);
                    filters[i][n][j][k] = init_val;
                }
            }
        }
    }

    out_dim = floor(((image_dim - filter_dim) / stride) + 1);
    vector<float> bias(num_filters, 0); // 1 bias per filter
}

Conv_Layer::Conv_Layer(){}

// dprev is gradient of loss function from one node forward (we're moving backwards here)
// do forward and backward pass for each image, calculate gradient for each image
// collect them all in each batch and then update weights and biases at the end of the batch
array3D<float> Conv_Layer::backward(array3D<float> &dprev, bool reset_grads)
{
    int dprev_dim = dprev[0].size();
    // should be num_filters channels in dprev
    int out_dim_f = floor(((image_dim - dprev_dim) / stride) + 1); // should be filter_dim of current layer
    int out_dim_x = dprev_dim + filter_dim - 1;                    // should be out_dim of previous layer relative to forward direction
    assert(out_dim_x == image_dim);

    // this should have same dimensions as input, this is not summed, fresh everytime
    vector<vector<vector<float> > > dx(num_channels, vector<vector<float>>(out_dim_x, vector<float>(out_dim_x, 0)));

    if(reset_grads) {        
        vector<vector<vector<float> > > df(num_filters, vector<vector<float>>(filter_dim, vector<float>(filter_dim, 0)));
        vector<float> dB(num_filters, 0); // 1 bias per filter
    }

    for (int f = 0; f < num_filters; f++)
    {
        for (int n = 0; n < num_channels; n++)
        {
            int curr_y = 0, out_y = 0;

            // calculate dL/dF
            while (curr_y + dprev_dim <= image_dim)
            {
                int curr_x = 0, out_x = 0;
                while (curr_x + dprev_dim <= image_dim)
                {
                    float sum = 0;
                    for (int kr = 0; kr < dprev_dim; kr++)
                    {
                        for (int kc = 0; kc < dprev_dim; kc++)
                        {
                            // dO/dF_ij = X_ij (local gradient)
                            // dL/dF_ij = dL/dprev_ij * X_ij (chain rule)
                            // dL/dF = conv(X, dL/dprev), normal convolution using only full overlap
                            float prod = dprev[f][kr][kc] * image[n][curr_y + kr][curr_x + kc];
                            sum += prod;
                        }
                    }
                    df[f][out_y][out_x] += sum; // += if multiple channels in image, = if 1 channel, would still sum if 1 channel in batch gd
                    curr_x += stride;
                    out_x++;
                }
                curr_y += stride;
                out_y++;
            }
            //print_matrices_df(image, dprev[f], df[f], out_dim_f);

            // calculate dL/dX
            array2D<float> orig_f = filters[f][n];
            //print_filter(orig_f);
            array2D<float> curr_f = rotate_180(orig_f);
            //print_filter(curr_f);
            curr_y = filter_dim - 1; // start on the bottom and move up
            out_y = 0;
            while (curr_y > -1 * dprev_dim)
            {
                int curr_x = filter_dim - 1; // start all the way to the right and move left
                int out_x = 0, conv_start_y = 0, conv_limit_y = 0, filt_start_y = 0;
                if (out_y < filter_dim)
                {
                    conv_start_y = 0;
                    conv_limit_y = out_y + 1;
                    if(conv_limit_y > dprev_dim)
                        conv_limit_y = dprev_dim;
                    filt_start_y = filter_dim - (out_y + 1);
                }
                else
                { // this means d_prev hanging off top, curr_y is negative
                    conv_start_y = -1 * curr_y;
                    conv_limit_y = dprev_dim;
                    filt_start_y = 0;
                }
                while (curr_x > -1 * dprev_dim)
                {
                    float sum = 0;
                    int conv_start_x = 0, conv_limit_x = 0, filt_start_x = 0;
                    if (out_x < filter_dim)
                    {
                        conv_start_x = 0;
                        conv_limit_x = out_x + 1;
                        if(conv_limit_x > dprev_dim)
                            conv_limit_x = dprev_dim;
                        filt_start_x = filter_dim - (out_x + 1);
                    }
                    else
                    { // this means dprev hanging off left side, curr_x is negative
                        conv_start_x = -1 * curr_x;
                        conv_limit_x = dprev_dim;
                        filt_start_x = 0; // if conv hanging off left side should always start at left most column
                    }
                    for (int dr = conv_start_y, fr = filt_start_y ; dr < conv_limit_y && fr < filter_dim; dr++, fr++)
                    {
                        for (int dc = conv_start_x, fc = filt_start_x ; dc < conv_limit_x && fc < filter_dim; dc++, fc++)
                        {
                            // dO/dF_ij = F_ij (local gradient)
                            // dL/dX_ij = dL/dprev_ij * F_ij (chain rule)
                            // dL/dX = conv(rot180(F), dL/dprev), full convolution
                            float prod = dprev[f][dr][dc] * curr_f[fr][fc];
                            sum += prod;
                        }
                    }
                    dx[n][out_y][out_x] += sum; // += if mulitple channels, = if one channel
                    curr_x -= stride;
                    out_x++;
                }
                curr_y -= stride;
                out_y++;
            }

            //cout << "Printing filter " << f << endl;
            //print_matrices_dx(curr_f, dprev[f], dx[f], out_dim_x);
        }
        array2D<float> dprev_channel = dprev[f];
        float dprev_sum = 0;
        for (int ii = 0; ii < dprev_dim; ii++)
            for (int jj = 0; jj < dprev_dim; jj++)
                dprev_sum += dprev_channel[ii][jj];
        dB[f] += dprev_sum;
    }
    return dx;
}

// use std::array if size is fixed at compile time, vector if not
array3D<float> Conv_Layer::forward(array3D<float> &image)
{
    vector<vector<vector<float>>> out(num_filters, vector<vector<float>>(out_dim, vector<float>(out_dim, 0)));
    for (int f = 0; f < num_filters; f++)
    {
        for (int n = 0; n < num_channels; n++)
        {        
            int curr_y = 0, out_y = 0;
            while (curr_y + filter_dim <= image_dim)
            {
                int curr_x = 0, out_x = 0;
                while (curr_x + filter_dim <= image_dim)
                {
                    float sum = 0;
                    for (int kr = 0; kr < filter_dim; kr++)
                    {
                        for (int kc = 0; kc < filter_dim; kc++)
                        {
                            float prod = filters[f][n][kr][kc] * image[n][curr_y + kr][curr_x + kc];
                            sum += prod;
                        }
                    }
                    out[f][out_y][out_x] += sum; // += if multiple channels in image, just = otherwise
                    if (n == 0)
                        out[f][out_y][out_x] += bias[f]; // only add bias once
                    curr_x += stride;
                    out_x++;
                }
                curr_y += stride;
                out_y++;                
            }
        }
    }
    return out;
}

int Conv_Layer::get_out_dim()
{
    return out_dim;
}

array3D<float>& Conv_Layer::get_dF() {
    return df;
}

vector<float>& Conv_Layer::get_dB() {
    return dB;
}