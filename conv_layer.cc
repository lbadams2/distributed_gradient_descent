#include "cnn.h"

Conv_Layer::Conv_Layer(int nf): num_filters(nf) {
    // filters initialized using standard normal, bias initialized all zeroes
    vector<vector<vector<double> > > filters(num_filters, vector<vector<double> >(FILTER_DIM, vector<double>(FILTER_DIM)));
    double stddev = 1 / sqrt(FILTER_DIM * FILTER_DIM);
    normal_dist = normal_distribution<double> distribution(0, stddev);
    default_random_engine generator;
    for(int i = 0; i < num_filters; i++) {
        for(int j = 0; j < FILTER_DIM; j++) {
            for(int k = 0; k < FILTER_DIM; k++) {
                double init_val = normal_dist(generator);
                filters[i][j][k] = init_val;
            }
        }
    }
    vector<double> bias(FILTER_DIM, 0);
}

// dprev is gradient of loss function from one node forward (we're moving backwards here)
// do forward and backward pass for each image, calculate gradient for each image
// collect them all in each batch and then update weights and biases at the end of the batch
void Conv_Layer::backward(array3D<double> dprev, array2D<int> image, int stride, array3D<double> &df, array3D<double> &dx) {
    int num_images = image.size();
    int dprev_dim = dprev[0].size();
    int image_dim = image.size();
    int out_dim_f = floor(((image_dim - dprev_dim) / stride) + 1); // 6
    int out_dim_x = dprev_dim + FILTER_DIM - 1;

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
                        double prod = dprev[f][kr][kc] * image[curr_y + kr][curr_x + kc];
                        sum += prod;
                    }
                }
                df[f][out_y][out_x] = sum;
                curr_x += stride;
                out_x++;
            }
            curr_y += stride;
            out_y++;
        }
        //print_matrices_df(image, dprev[f], df[f], out_dim_f);

        // calculate dL/dX
        array2D<int> orig_f = filters[f];
        //print_filter(orig_f);
        array2D<int> curr_f = rotate_180(orig_f);
        //print_filter(curr_f);
        curr_y = FILTER_DIM - 1; // start on the bottom and move up
        out_y = 0;
        while (curr_y > -1 * dprev_dim)
        {
            int curr_x = FILTER_DIM - 1; // start all the way to the right and move left
            int out_x = 0, conv_start_y = 0, conv_limit_y = 0, filt_start_y = 0;
            if(out_y < FILTER_DIM) {
                conv_start_y = 0;
                conv_limit_y = out_y + 1;
                filt_start_y = FILTER_DIM - (out_y + 1);
            }
            else { // this means d_prev hanging off top, curr_y is negative
                conv_start_y = -1 * curr_y;
                conv_limit_y = FILTER_DIM;
                filt_start_y = 0;
            }
            while (curr_x > -1 * dprev_dim)
            {
                double sum = 0;
                int conv_start_x = 0, conv_limit_x = 0, filt_start_x = 0;
                if(out_x < FILTER_DIM) {
                    conv_start_x = 0;
                    conv_limit_x = out_x + 1;
                    filt_start_x = FILTER_DIM - (out_x + 1);
                }
                else { // this means dprev hanging off left side, curr_x is negative
                    conv_start_x = -1 * curr_x;
                    conv_limit_x = FILTER_DIM;
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
                dx[f][out_y][out_x] = sum;
                curr_x -= stride;
                out_x++;
            }
            curr_y -= stride;
            out_y++;
        }

        //cout << "Printing filter " << f << endl;
        //print_matrices_dx(curr_f, dprev[f], dx[f], out_dim_x);
    }
}

// use std::array if size is fixed at compile time, vector if not
array3D<double> Conv_Layer::forward(array2D<int> image, int stride) {
    int out_dim = floor(((IMAGE_DIM - FILTER_DIM) / stride) + 1);
    vector<vector<vector<double> > > out(num_filters, vector<vector<double> >(out_dim, vector<double>(out_dim)));
    for(int f = 0; f < num_filters; f++) {
        int curr_y, out_y = 0;
        while(curr_y + FILTER_DIM <= IMAGE_DIM) {
            int curr_x, out_x = 0;
            while(curr_x + FILTER_DIM <= IMAGE_DIM) {
                double sum = 0;
                for (int kr = 0; kr < FILTER_DIM; kr++)
                {
                    for (int kc = 0; kc < FILTER_DIM; kc++)
                    {
                        double prod = filters[f][kr][kc] * image[curr_y + kr][curr_x + kc];
                        sum += prod;
                    }
                }
                out[f][out_y][out_x] = sum;
                curr_x += stride;
                out_x++;
            }
            curr_y += stride;
            out_y++;
        }
    }
    return out;
}