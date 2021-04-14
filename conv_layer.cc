#include "cnn.h"

Conv_Layer::Conv_Layer(int nf, int filter_dim, int stride): num_filters(nf), filter_dim(filter_dim), stride(stride) {
    // filters initialized using standard normal, bias initialized all zeroes
    vector<vector<vector<double> > > filters(num_filters, vector<vector<double> >(filter_dim, vector<double>(filter_dim)));
    double stddev = 1 / sqrt(filter_dim * filter_dim);
    normal_dist = normal_distribution<double> distribution(0, stddev);
    default_random_engine generator;
    for(int i = 0; i < num_filters; i++) {
        for(int j = 0; j < filter_dim; j++) {
            for(int k = 0; k < filter_dim; k++) {
                double init_val = normal_dist(generator);
                filters[i][j][k] = init_val;
            }
        }
    }
    vector<double> bias(num_filters, 0); // 1 bias per filter
    vector<vector<vector<double> > > df(num_filters, vector<vector<double> >(filter_dim, vector<double>(filter_dim, 0)));
}

// dprev is gradient of loss function from one node forward (we're moving backwards here)
// do forward and backward pass for each image, calculate gradient for each image
// collect them all in each batch and then update weights and biases at the end of the batch
array3D<double> Conv_Layer::backward(array3D<double> &dprev) {
    int num_channels = image.size();
    int dprev_dim = dprev[0].size();
    // should be num_filters channels in dprev
    int image_dim = image[0].size();
    int out_dim_f = floor(((image_dim - dprev_dim) / stride) + 1); // should be filter_dim of current layer
    int out_dim_x = dprev_dim + filter_dim - 1; // should be out_dim of previous layer relative to forward direction
    vector<vector<vector<double> > > dx(num_filters, vector<vector<double> >(out_dim_x, vector<double>(out_dim_x, 0)));
    
    for(int n = 0; n < num_channels; n++) {
        for (int f = 0; f < num_filters; f++)
        {
            int curr_y = 0, out_y = 0;

            // calculate dL/dF
            while (curr_y + dprev_dim <= image_dim)
            {
                int curr_x = 0, out_x = 0;
                while (curr_x + dprev_dim <= image_dim)
                {
                    double sum = 0;
                    for (int kr = 0; kr < dprev_dim; kr++)
                    {
                        for (int kc = 0; kc < dprev_dim; kc++)
                        {
                            // dO/dF_ij = X_ij (local gradient)
                            // dL/dF_ij = dL/dprev_ij * X_ij (chain rule)
                            // dL/dF = conv(X, dL/dprev), normal convolution using only full overlap
                            double prod = dprev[f][kr][kc] * image[n][curr_y + kr][curr_x + kc];
                            sum += prod;
                        }
                    }
                    df[f][out_y][out_x] += sum; // += if multiple channels in image, = if 1 channel
                    curr_x += stride;
                    out_x++;
                }
                curr_y += stride;
                out_y++;
            }
            //print_matrices_df(image, dprev[f], df[f], out_dim_f);

            // calculate dL/dX
            array2D<double> orig_f = filters[f];
            //print_filter(orig_f);
            array2D<double> curr_f = rotate_180(orig_f);
            //print_filter(curr_f);
            curr_y = filter_dim - 1; // start on the bottom and move up
            out_y = 0;
            while (curr_y > -1 * dprev_dim)
            {
                int curr_x = filter_dim - 1; // start all the way to the right and move left
                int out_x = 0, conv_start_y = 0, conv_limit_y = 0, filt_start_y = 0;
                if(out_y < filter_dim) {
                    conv_start_y = 0;
                    conv_limit_y = out_y + 1;
                    filt_start_y = filter_dim - (out_y + 1);
                }
                else { // this means d_prev hanging off top, curr_y is negative
                    conv_start_y = -1 * curr_y;
                    conv_limit_y = filter_dim;
                    filt_start_y = 0;
                }
                while (curr_x > -1 * dprev_dim)
                {
                    double sum = 0;
                    int conv_start_x = 0, conv_limit_x = 0, filt_start_x = 0;
                    if(out_x < filter_dim) {
                        conv_start_x = 0;
                        conv_limit_x = out_x + 1;
                        filt_start_x = filter_dim - (out_x + 1);
                    }
                    else { // this means dprev hanging off left side, curr_x is negative
                        conv_start_x = -1 * curr_x;
                        conv_limit_x = filter_dim;
                        filt_start_x = 0; // if conv hanging off left side should always start at left most column
                    }
                    for (int dr = conv_start_y, fr = filt_start_y ; dr < conv_limit_y; dr++, fr++)
                    {
                        for (int dc = conv_start_x, fc = filt_start_x ; dc < conv_limit_x; dc++, fc++)
                        {
                            // dO/dF_ij = F_ij (local gradient)
                            // dL/dX_ij = dL/dprev_ij * F_ij (chain rule)   
                            // dL/dX = conv(rot180(F), dL/dprev), full convolution
                            double prod = dprev[f][dr][dc] * curr_f[fr][fc];
                            sum += prod;
                        }
                    }
                    dx[f][out_y][out_x] += sum; // += if mulitple channels, = if one channel
                    curr_x -= stride;
                    out_x++;
                }
                curr_y -= stride;
                out_y++;
            }

            //cout << "Printing filter " << f << endl;
            //print_matrices_dx(curr_f, dprev[f], dx[f], out_dim_x);

            if(n == 0) {
                array2D<double> dprev_channel = dprev[f];
                int dprev_sum = 0;
                for(int ii = 0; ii < dprev_dim; ii++)
                    for(int jj = 0; jj < dprev_dim; jj++)
                        dprev_sum += dprev_channel[ii][jj];
                dB[f] = dprev_sum;
            }
        }
    }
    return dx;
}

// use std::array if size is fixed at compile time, vector if not
array3D<double> Conv_Layer::forward(array3D<double> &image) {
    int out_dim = floor(((IMAGE_DIM - filter_dim) / stride) + 1);
    vector<vector<vector<double> > > out(num_filters, vector<vector<double> >(out_dim, vector<double>(out_dim, 0)));
    int num_channels = image.size();
    for(int n = 0; n < num_channels; n++) {
        for(int f = 0; f < num_filters; f++) {
            int curr_y, out_y = 0;
            while(curr_y + filter_dim <= IMAGE_DIM) {
                int curr_x, out_x = 0;
                while(curr_x + filter_dim <= IMAGE_DIM) {
                    double sum = 0;
                    for (int kr = 0; kr < filter_dim; kr++)
                    {
                        for (int kc = 0; kc < filter_dim; kc++)
                        {
                            double prod = filters[f][kr][kc] * image[n][curr_y + kr][curr_x + kc];
                            sum += prod;
                        }
                    }
                    out[f][out_y][out_x] += sum; // += if multiple channels in image, just = otherwise
                    curr_x += stride;
                    out_x++;
                }
                curr_y += stride;
                out_y++;
                if(n == 0)
                    out[f][out_y][out_x] += bias[f]; // only add bias once
            }            
        }
    }
    return out;
}