#include "cnn.h"

array3D<double> MaxPool_Layer::forward(array3D<double> &image) {
    int num_channels = image.size();
    int orig_dim = image[0].size();
    int new_dim = ((orig_dim - kernel_size) / stride) + 1;
    vector<vector<vector<double> > > downsampled(num_channels, vector<vector<double> >(new_dim, vector<double>(new_dim)));
    for(int n = 0; n < num_channels; n++) {
        int curr_y = 0, out_y = 0;
        while(curr_y + kernel_size <= orig_dim) {
            int curr_x = 0, out_x = 0;
            while(curr_x + kernel_size <= orig_dim) {
                double max = std::numeric_limits<double>::min();
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
    double max_val = std::numeric_limits<double>::min();
    double nan = std::numeric_limits<double>::max();
    for(int i = curr_y; i < curr_y + kernel_size; i++) {
        vector<double> row = orig_image[channel_num][i];
        for(int j = curr_x; j < curr_x + kernel_size; j++) {
            if(row[j] > max_val && row[j] != nan) {
                y_max = i;
                x_max = j;
                max_val = row[j];
            }
        }
    }
}

array3D<double> MaxPool_Layer::backward(array3D<double> &dprev) {
    int orig_dim = orig_image[0].size();
    int num_channels = orig_image.size();
    int curr_y = 0, out_y = 0;
    int y_max = -1, x_max = -1;
    vector<vector<vector<double> > > dout(num_channels, vector<vector<double> >(orig_dim, vector<double>(orig_dim)));
    for(int n = 0; n < num_channels; n++) {
        while(curr_y + kernel_size <= orig_dim) {
            int curr_x = 0, out_x = 0;
            while(curr_x + kernel_size <= orig_dim) {
                argmax(n, curr_y, curr_x, y_max, x_max); // find index of value that was chosen by max pool in forward pass
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