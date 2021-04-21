#include "cnn.h"

MaxPool_Layer::MaxPool_Layer(int kernel_size, int stride) : kernel_size(kernel_size), stride(stride) {

}

array3D<float> MaxPool_Layer::forward(array3D<float> &image) {
    int num_channels = image.size();
    int orig_dim = image[0].size();
    int new_dim = ((orig_dim - kernel_size) / stride) + 1;
    this->orig_image = image;
    vector<vector<vector<float> > > downsampled(num_channels, vector<vector<float> >(new_dim, vector<float>(new_dim)));
    for(int n = 0; n < num_channels; n++) {
        int curr_y = 0, out_y = 0;
        while(curr_y + kernel_size <= orig_dim) {
            int curr_x = 0, out_x = 0;
            while(curr_x + kernel_size <= orig_dim) {
                float max = std::numeric_limits<float>::min();
                for(int r = curr_y; r < curr_y + kernel_size; r++) {
                    for(int c = curr_x; c < curr_x + kernel_size; c++) {
                        if(image[n][r][c] > max)
                            max = image[n][r][c];
                        downsampled[n][out_y][out_x] = max;
                    }
                }
                curr_x += stride;
                out_x += 1;
            }
            curr_y += stride;
            out_y += 1;
        }
    }
    return downsampled;
}

void MaxPool_Layer::argmax(int channel_num, int curr_y, int curr_x, int &y_max, int &x_max) {
    float nan = std::numeric_limits<float>::max();
    float max_val = -1 * nan;
    for(int i = curr_y; i < curr_y + kernel_size; i++) {
        vector<float> row = orig_image[channel_num][i];
        for(int j = curr_x; j < curr_x + kernel_size; j++) {
            if(row[j] > max_val && row[j] != nan) {
                y_max = i;
                x_max = j;
                max_val = row[j];
            }
        }
    }
}

array3D<float> MaxPool_Layer::backward(array3D<float> &dprev) {
    int orig_dim = orig_image[0].size();
    int num_channels = orig_image.size();
    int curr_y = 0, out_y = 0;
    int y_max = -1, x_max = -1;
    vector<vector<vector<float> > > dout(num_channels, vector<vector<float> >(orig_dim, vector<float>(orig_dim)));
    for(int n = 0; n < num_channels; n++) {
        while(curr_y + kernel_size <= orig_dim) {
            int curr_x = 0, out_x = 0;
            while(curr_x + kernel_size <= orig_dim) {
                argmax(n, curr_y, curr_x, y_max, x_max); // find index of value that was chosen by max pool in forward pass
                // argmax should return max val for current window, window moves in each inner and outer while, each iteration has its own max
                // that should be set in dout
                dout[n][y_max][x_max] = dprev[n][out_y][out_x]; // only non zero values will be at indexes chosen in forward pass
                curr_x += stride;
                out_x++;
            }
            curr_y += stride;
            out_y++;
        }
    }
    return dout;
}

int MaxPool_Layer::get_out_dim(int conv_dim) {
    int pool_output_dim = ((conv_dim - kernel_size) / stride) + 1;
    return pool_output_dim;
}
