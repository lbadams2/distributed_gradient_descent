#include "cnn.h"

Conv_Layer::Conv_Layer(int nf): num_filters(nf) {
    // filters initialized using standard normal, bias initialized all zeroes
    filters = array<array<array<double, out_dim>, out_dim>, num_filters>();
    stddev = 1 / sqrt(FILTER_DIM * FILTER_DIM);
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
    bias = array<double, FILTER_DIM>();
    for(int i = 0; i < FILTER_DIM; i++)
        bias[i] = 0;
}

Dense_Layer::Dense_Layer(int in_dim, int out_dim): in_dim(in_dim), out_dim(out_dim) {
    // init weights and bias
    weights = array<array<double, out_dim>, in_dim>;
    normal_dist = normal_distribution<double> distribution(0, 1);
    default_random_engine generator;
    for(int i = 0; i < in_dim; i++) {
        for(int j = 0; j < out_dim; j++) {
            double init_val = normal_dist(generator);
            weights[i][j] = init_val * .01;
        }
    }
    bias = array<double, out_dim>();
    for(int i = 0; i < out_dim; i++)
        bias[i] = 0;
}

// use std::array if size is fixed at compile time, vector if not
array3D Conv_Layer::forward(array2D image, int stride) {
    out_dim = floor(((IMAGE_DIM - FILTER_DIM) / stride) + 1);
    array3D out = array<array<array<double, out_dim>, out_dim>, num_filters>();
    for(int i = 0; i < num_filters; i++) {
        int curr_y, out_y = 0; // location of 
        array2D curr_f = array3D[i];
        while(curr_y + FILTER_DIM <= IMAGE_DIM) {
            int curr_x, out_x = 0;
            while(curr_x + FILTER_DIM <= IMAGE_DIM) {
                double curr_sum = 0;
                for(int r = curr_y, int f_r = 0; r < curr_y + FILTER_DIM; r++, f_r++) {
                    for(int c = curr_x, int f_c = 0; c < curr_x + FILTER_DIM; c++, f_c++) {
                        double prod = image[r][c] * curr_f[f_r][f_c];
                        curr_sum += prod;
                        curr_sum += bias[i];
                    }
                }
                out[i][out_y][out_x] = curr_sum;
                curr_x += stride;
                out_x++;
            }
            curr_y += stride;
            out_y++;
        }
    }
    return out;
}

array2D maxpool(array2D image, int kernel_size, int stride) {
    new_dim = ((IMAGE_DIM - kernel_size) / stride) + 1;
    array2D downsampled = array<array<double, new_dim>, new_dim>();
    int curr_y, out_y = 0;
    while(curr_y + kernel_size <= IMAGE_DIM) {
        int curr_x, out_x = 0;
        while(curr_x + kernel_size <= IMAGE_DIM) {
            int max = 0;
            for(int r = curr_y, int f_r = 0; r < curr_y + FILTER_DIM; r++, f_r++) {
                for(int c = curr_x, int f_c = 0; c < curr_x + FILTER_DIM; c++, f_c++) {
                    if(image[r][c] > max)
                        max = image[r][c];
                    downsampled[out_y][out_x] = max;
                    curr_x += stride;
                    out_x += 1;
                }
                curr_y += stride;
                out_y += 1;
            }
        }
    }
    return downsampled;
}

array flatten(array2D image) {
    rows = image.size();
    cols = image[0].size();
    flattened_dim = rows * cols;
    array flattened = array<double, flattened_dim>();
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++) {
            k = (i * cols) + j;
            flattened[k] = image[i][j];
        }
    return flattened;
}

void relu(array in) {
    for( int &p : in )
        if(p < 0)
            p = 0;
}

void softmax(array in) {
    double sum = 0;
    for( int &p : in ) {
        exp(p);
        sum += p;
    }
    for( int &p : in )
        p /= sum;
}

double cat_cross_entropy(pred_probs, true_labels) {
    int i, tmp = 0;
    double sum = 0;
    for( int &p : pred_probs) {
        l = true_labels[i];
        tmp = l * log(p);
        sum += tmp;
        i++;
    }
    return -sum;
}

array Dense_Layer::forward(array in) {
    rows = weights.size();
    cols = weights[0].size();
    array product = array<double, rows>();
    int tmp = 0;
    for(int i = 0; i < rows; i++) {
        double out_value = 0;
        for(int j = 0; j < cols; j++) {
            tmp = weights[i][j] * in[j];
            out_value += tmp;
        }
        product[i] = out_value;
    }

    // ReLU
    relu(product);
    return product;
}