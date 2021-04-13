#include "cnn.h"

array2D<double> MaxPool_Layer::forward(array2D<double> &image) {
    int new_dim = ((IMAGE_DIM - kernel_size) / stride) + 1;
    vector<vector<double> > downsampled(new_dim, vector<double>(new_dim));
    int curr_y, out_y = 0;
    while(curr_y + kernel_size <= IMAGE_DIM) {
        int curr_x, out_x = 0;
        while(curr_x + kernel_size <= IMAGE_DIM) {
            int max = 0;
            for(int r = curr_y; r < curr_y + FILTER_DIM; r++) {
                for(int c = curr_x; c < curr_x + FILTER_DIM; c++) {
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

void MaxPool_Layer::argmax(int curr_y, int curr_x, int &y_max, int &x_max) {
    double max_val = std::numeric_limits<double>::min();
    double nan = std::numeric_limits<double>::max();
    for(int i = curr_y; i < curr_y + kernel_size; i++) {
        vector<double> row = orig_image[i];
        for(int j = curr_x; j < curr_x + kernel_size; j++) {
            if(row[j] > max_val && row[j] != nan) {
                y_max = i;
                x_max = j;
                max_val = row[j];
            }
        }
    }
}

array2D<double> MaxPool_Layer::backward(array2D<double> &dprev) {
    int orig_dim = orig_image.size();
    int curr_y = 0, out_y = 0;
    int y_max = -1, x_max = -1;
    vector<vector<double> > dout(orig_dim, vector<double>(orig_dim, 0));
    while(curr_y + kernel_size <= orig_dim) {
        int curr_x = 0, out_x = 0;
        while(curr_x + kernel_size <= orig_dim) {
            argmax(curr_y, curr_x, y_max, x_max); // find index of value that was chosen by max pool in forward pass
            dout[y_max][x_max] = dprev[out_y][out_x]; // only non zero values will be at indexes chosen in forward pass
            curr_x += stride;
            out_x++;
        }
        curr_y += stride;
        out_y++;
    }
    return dout;
}