//#include "opencv2/highgui.hpp"
#include <vector>
#include <array>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include <assert.h>

using std::array;
using std::cout;
using std::endl;
using std::reverse;
using std::vector;
using std::min;
using std::string;
using std::ifstream;
using std::ios;
using std::runtime_error;
using std::sqrt;

typedef unsigned char uchar;

template<typename T>
using array2D = vector<vector<T> >;

template<typename T>
using array3D = vector<vector<vector<T> > >;

template<typename T>
using array4D = vector<vector<vector<vector<T> > > >;

#define NUM_CHANNELS 3
#define IMAGE_DIM 28
#define FILTER_DIM 5

void print_matrices_conv(array2D<int> image, array2D<int> filter, array2D<int> conv, int out_dim)
{
    cout << "printing image" << endl;
    for (int i = 0; i < IMAGE_DIM; i++)
    {
        cout << endl;
        for (int j = 0; j < IMAGE_DIM; j++)
            cout << image[i][j] << " ";
    }

    cout << "\n\n";
    cout << "printing filter" << endl;
    for (int i = 0; i < FILTER_DIM; i++)
    {
        cout << endl;
        for (int j = 0; j < FILTER_DIM; j++)
            cout << filter[i][j] << " ";
    }

    cout << "\n\n";
    cout << "printing output" << endl;
    for (int i = 0; i < out_dim; i++)
    {
        cout << endl;
        for (int j = 0; j < out_dim; j++)
            cout << conv[i][j] << " ";
    }
    cout << "\n\n";
}

void print_matrices_df(array2D<int> image, array2D<int> dprev, array2D<int> df, int out_dim)
{
    cout << "printing image" << endl;
    int image_dim = image.size(); // may be downsampled
    for (int i = 0; i < image_dim; i++)
    {
        cout << endl;
        for (int j = 0; j < image_dim; j++)
            cout << image[i][j] << " ";
    }

    cout << "\n\n";
    cout << "printing dprev" << endl;
    int dprev_dim = dprev.size();
    for (int i = 0; i < dprev_dim; i++)
    {
        cout << endl;
        for (int j = 0; j < dprev_dim; j++)
            cout << dprev[i][j] << " ";
    }

    cout << "\n\n";
    cout << "printing output" << endl;
    for (int i = 0; i < out_dim; i++)
    {
        cout << endl;
        for (int j = 0; j < out_dim; j++)
            cout << df[i][j] << " ";
    }
    cout << "\n\n";
}

void print_matrices_dx(array2D<int> filter, array2D<int> dprev, array2D<int> dx, int out_dim)
{
    cout << "printing filter" << endl;
    int filter_dim = filter.size(); // may be downsampled
    for (int i = 0; i < filter_dim; i++)
    {
        cout << endl;
        for (int j = 0; j < filter_dim; j++)
            cout << filter[i][j] << " ";
    }

    cout << "\n\n";
    cout << "printing dprev" << endl;
    int dprev_dim = dprev.size();
    for (int i = 0; i < dprev_dim; i++)
    {
        cout << endl;
        for (int j = 0; j < dprev_dim; j++)
            cout << dprev[i][j] << " ";
    }

    cout << "\n\n";
    cout << "printing output" << endl;
    for (int i = 0; i < out_dim; i++)
    {
        cout << endl;
        for (int j = 0; j < out_dim; j++)
            cout << dx[i][j] << " ";
    }
    cout << "\n\n";
}

void print_matrix(array2D<int> &in) {
    for(vector<int> row : in) {
        cout << endl;
        for(int val : row) {
            cout << val << " ";
        }
    }
}

// value at top left corner of kernel multiplied by value at bottom right of neighborhood
// size of std::array cannot be set with variable at runtime

// input to first conv layer is 1 x 28 x 28, filters are num_filters_first x 1 x filter_dim x filter_dim
// out_dim = floor(((IMAGE_DIM - filter_dim) / stride) + 1);
// output will be num_filters x out_dim x out_dim

// input to second conv layer is num_filters x out_dim x out_dim, filters are num_filters_second x num_filters_first x filter_dim x filter_dim
// the channels of the filters should match the channels of the input

// filter channels match image channels, each filter channel responsible for one image channel
// when a filter with n channels is applied to an image with n channels, only the ith filter channel is convolved with the ith image channel (no crossing)
// then the n convolutions are summed to produce 1 x out_dim x out_dim output, the final output of the layer appends the output of each filter
// so final output is num_filters x out_dim x out_dim
array3D<int> convolution(array3D<int> &image, array4D<int> filters, vector<int> &bias, int stride) {
    int image_dim = image[0].size();
    int num_filters = filters.size();
    int filter_dim = filters[0][0].size();
    const int out_dim = floor(((image_dim - filter_dim) / stride) + 1);
    vector<vector<vector<int> > > out(num_filters, vector<vector<int> >(out_dim, vector<int>(out_dim, 0)));
    int num_channels = image.size();
    int filter_channels = filters[0].size();
    assert(num_channels == filter_channels);
    for(int f = 0; f < num_filters; f++) {
        for(int n = 0; n < num_channels; n++) {
            int curr_y = 0, out_y = 0;
            while(curr_y + filter_dim <= image_dim) {
                int curr_x = 0, out_x = 0;
                while(curr_x + filter_dim <= image_dim) {
                    double sum = 0;
                    for (int kr = 0; kr < filter_dim; kr++)
                    {
                        for (int kc = 0; kc < filter_dim; kc++)
                        {
                            double prod = filters[f][n][kr][kc] * image[n][curr_y + kr][curr_x + kc];
                            sum += prod;
                        }
                    }
                    out[f][out_y][out_x] += sum; // += if multiple channels in image, just = otherwise
                    if(n == 0) {
                        //cout << "bias added filter " << f << " channel " << n << endl;
                        out[f][out_y][out_x] += bias[f]; // only add bias once
                    }
                    curr_x += stride;
                    out_x++;
                }
                curr_y += stride;
                out_y++;
            }
            // print output
            if(n == 0) {
                cout << "printing channel " << n << endl;
                cout << "image channel " << n << endl;
                print_matrix(image[n]);
                cout << endl;
                cout << "filter " << f << endl;
                print_matrix(filters[f][n]);
                cout << endl;
                cout << "out " << f << endl;
                print_matrix(out[f]);
                cout << endl;
                cout << endl;
            }
            else if(n == 1) {
                cout << "printing channel 1, should be sum of channel 0 and channel 1" << endl;
                cout << "image channel " << n << endl;
                print_matrix(image[n]);
                cout << endl;
                cout << "filter " << f << endl;
                print_matrix(filters[f][n]);
                cout << endl;
                cout << "out " << f << endl;
                print_matrix(out[f]);
                cout << endl;
                cout << endl;
            }
        }
    }
    return out;
}

array2D<int> rotate_180(array2D<int> filter)
{
    reverse(std::begin(filter), std::end(filter)); // reverse rows
    std::for_each(std::begin(filter), std::end(filter),
                  [](auto &i) { reverse(std::begin(i), std::end(i)); }); // reverse columns
    return filter;
}

void print_filter(array2D<int> filter) {
    cout << "printing filter" << endl;
    for (int i = 0; i < FILTER_DIM; i++)
    {
        cout << endl;
        for (int j = 0; j < FILTER_DIM; j++)
            cout << filter[i][j] << " ";
    }
    cout << endl;
}

void conv_back(array3D<int> dprev, array4D<int> filters, array3D<int> image, int stride, array3D<int> &df, array3D<int> &dx, vector<int> &dB)
{
    int num_filters = filters.size();
    int filter_dim = filters[0][0].size();
    int dprev_dim = dprev[0].size();
    int image_dim = image[0].size();
    
    int out_dim_f = floor(((image_dim - dprev_dim) / stride) + 1); // 6
    assert(out_dim_f == filter_dim);
    
    int out_dim_x = dprev_dim + filter_dim - 1;
    assert(out_dim_x == image_dim);

    int filter_channels = filters[0].size();
    int image_channels = image.size();
    assert(filter_channels == image_channels);

    for (int f = 0; f < num_filters; f++)
    {
        for(int n = 0; n < filter_channels; n++) {
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
                    df[f][out_y][out_x] += sum; // sum over all channels
                    curr_x += stride;
                    out_x++;
                }
                curr_y += stride;
                out_y++;                
            } // end dL/dF while
            
            // print output
            /*
            if(n == 0) {
                cout << "printing channel " << n << endl;
                cout << "image channel " << n << endl;
                print_matrix(image[n]);
                cout << endl;
                cout << "dprev filter " << f << endl;
                print_matrix(dprev[f]);
                cout << endl;
                cout << "df " << f << endl;
                print_matrix(df[f]);
                cout << endl;
                cout << endl;
            }
            else if(n == 1) {
                cout << "printing channel 1, should be sum of channel 0 and channel 1" << endl;
                cout << "image channel " << n << endl;
                print_matrix(image[n]);
                cout << endl;
                cout << "dprev filter " << f << endl;
                print_matrix(dprev[f]);
                cout << endl;
                cout << "df " << f << endl;
                print_matrix(df[f]);
                cout << endl;
                cout << endl;
            }
            */

            // calculate dL/dX
            array2D<int> orig_f = filters[f][n];
            print_filter(orig_f);
            array2D<int> curr_f = rotate_180(orig_f);
            print_filter(curr_f);
            curr_y = filter_dim - 1; // start on the bottom and move up
            out_y = 0;
            while (curr_y > -1 * dprev_dim)
            {
                int curr_x = filter_dim - 1; // start all the way to the right and move left
                int out_x = 0, conv_start_y = 0, conv_limit_y = 0, filt_start_y = 0;
                if(out_y < filter_dim) {
                    conv_start_y = 0;
                    conv_limit_y = out_y + 1;
                    if(conv_limit_y > dprev_dim)
                        conv_limit_y = dprev_dim;
                    filt_start_y = filter_dim - (out_y + 1);
                }
                else { // this means d_prev hanging off top, curr_y is negative
                    conv_start_y = -1 * curr_y;
                    conv_limit_y = dprev_dim;
                    filt_start_y = 0;
                }
                while (curr_x > -1 * dprev_dim)
                {
                    double sum = 0;
                    int conv_start_x = 0, conv_limit_x = 0, filt_start_x = 0;
                    if(out_x < filter_dim) {
                        conv_start_x = 0;
                        conv_limit_x = out_x + 1;
                        if(conv_limit_x > dprev_dim)
                            conv_limit_x = dprev_dim;
                        filt_start_x = filter_dim - (out_x + 1);
                    }
                    else { // this means dprev hanging off left side, curr_x is negative
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
                            double prod = dprev[f][dr][dc] * curr_f[fr][fc];
                            sum += prod;
                        }
                    }
                    dx[n][out_y][out_x] += sum; // each channel is sum of all filters
                    curr_x -= stride;
                    out_x++;
                }
                curr_y -= stride;
                out_y++;
            } // end dL/dX while

            if(f == 0) {
                cout << "printing filter " << f << endl;
                cout << "filter num " << f << endl;
                print_matrix(curr_f);
                cout << endl;
                cout << "dprev filter " << f << endl;
                print_matrix(dprev[f]);
                cout << endl;
                cout << "dx " << n << endl;
                print_matrix(dx[n]);
                cout << endl;
                cout << endl;
            }
            else if(f == 1) {
                cout << "printing filter 1, should be sum of filter 0 and filter 1" << endl;
                cout << "filter num " << f << endl;
                print_matrix(curr_f);
                cout << endl;
                cout << "dprev filter " << f << endl;
                print_matrix(dprev[f]);
                cout << endl;
                cout << "dx " << n << endl;
                print_matrix(dx[n]);
                cout << endl;
                cout << endl;
            }            
        } // end for channels
        array2D<int> dprev_channel = dprev[f];
        int dprev_sum = 0;
        for (int ii = 0; ii < dprev_dim; ii++)
            for (int jj = 0; jj < dprev_dim; jj++)
                dprev_sum += dprev_channel[ii][jj];
        dB[f] = dprev_sum;
    } // end for filters
}

array3D<int> maxpool_forward(array3D<int> &image, int pool_dim, int stride) {
    int num_channels = image.size();
    int orig_dim = image[0].size();
    int new_dim = ((orig_dim - pool_dim) / stride) + 1;
    vector<vector<vector<int> > > downsampled(num_channels, vector<vector<int> >(new_dim, vector<int>(new_dim)));
    for(int n = 0; n < num_channels; n++) {
        int curr_y = 0, out_y = 0;
        while(curr_y + pool_dim <= orig_dim) {
            int curr_x = 0, out_x = 0;
            while(curr_x + pool_dim <= orig_dim) {
                int max = std::numeric_limits<int>::min();
                for(int r = curr_y; r < curr_y + pool_dim; r++) {
                    for(int c = curr_x; c < curr_x + pool_dim; c++) {
                        if(image[n][r][c] > max)
                            max = image[n][r][c];
                        downsampled[n][out_y][out_x] = max;
                    }
                }
                curr_x += stride;
                out_x++;
            }
            curr_y += stride;
            out_y++;
        }
    }
    print_matrix(image[0]);
    cout << endl;
    print_matrix(downsampled[0]);
    return downsampled;
}

void argmax(array3D<int> &orig_image, int channel_num, int curr_y, int curr_x, int pool_dim, int &y_max, int &x_max) {
    int max_val = std::numeric_limits<int>::min();
    int nan = std::numeric_limits<int>::max();
    for(int i = curr_y; i < curr_y + pool_dim; i++) {
        vector<int> row = orig_image[channel_num][i];
        for(int j = curr_x; j < curr_x + pool_dim; j++) {
            if(row[j] > max_val && row[j] != nan) {
                y_max = i;
                x_max = j;
                max_val = row[j];
            }
        }
    }
}

array3D<int> maxpool_backward(array3D<int> &dprev, array3D<int> &orig_image, int pool_dim, int stride) {
    int orig_dim = orig_image[0].size();
    int num_channels = orig_image.size();
    int curr_y = 0, out_y = 0;
    int y_max = -1, x_max = -1;
    vector<vector<vector<int> > > dout(num_channels, vector<vector<int> >(orig_dim, vector<int>(orig_dim)));
    for(int n = 0; n < num_channels; n++) {
        while(curr_y + pool_dim <= orig_dim) {
            int curr_x = 0, out_x = 0;
            while(curr_x + pool_dim <= orig_dim) {
                argmax(orig_image, n, curr_y, curr_x, pool_dim, y_max, x_max); // find index of value that was chosen by max pool in forward pass
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

vector<int> flatten(array3D<int> &image) {
    int rows = image[0].size();
    int cols = image[0][0].size();
    int channels = image.size();
    int flattened_dim = channels * rows * cols;
    vector<int> flattened(flattened_dim);
    for(int n = 0; n < channels; n++) {
        for(int i = 0; i < rows; i++) {
            int k = n * rows * cols; // offset of current square
            for(int j = 0; j < cols; j++) {
                int l = (i * cols) + j; // offset within square
                int offset = k + l;
                flattened[offset] = image[n][i][j];
            }
        }
    }
    return flattened;
}

array3D<int> unflatten(vector<int> &vec, int num_filters, int pool_output_dim) {
    vector<vector<vector<int> > > d_pool(num_filters, vector<vector<int> >(pool_output_dim, vector<int>(pool_output_dim)));
    int row = 0, col = 0, channel = 0, channel_offset = 0;
    for(int i = 0; i < vec.size(); i++) {
        channel = floor(i / (pool_output_dim * pool_output_dim));
        channel_offset = channel * pool_output_dim * pool_output_dim;
        row = floor((i - channel_offset) / pool_output_dim);
        col = (i - channel_offset) % pool_output_dim;
        d_pool[channel][row][col] = vec[i];
    }
    return d_pool;
}

void create_data_forward(array3D<int> &image, array4D<int> &filters, int num_filters, int num_channels)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(1, 10);

    for (int i = 0; i < num_channels; i++)
    {
        int val = i * 2;
        for (int r = 0; r < IMAGE_DIM; r++)
            for (int c = 0; c < IMAGE_DIM; c++)
                image[i][r][c] = val++;
    }

    for (int i = 0; i < num_filters; i++)
    {
        for(int n = 0; n <num_channels; n++) {
            int val = i * 2;
            for (int r = 0; r < FILTER_DIM; r++)
                for (int c = 0; c < FILTER_DIM; c++)
                    filters[i][n][r][c] = val++;
        }
    }
}

void create_data_back(array3D<int> &image, array4D<int> &filters, array3D<int> &dprev, int num_filters)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(1, 10);

    for (int i = 0; i < NUM_CHANNELS; i++)
    {
        int val = i * 2;
        for (int r = 0; r < IMAGE_DIM; r++)
            for (int c = 0; c < IMAGE_DIM; c++)
                image[i][r][c] = val++;
    }

    for (int i = 0; i < num_filters; i++)
    {
        for(int n = 0; n < NUM_CHANNELS; n++) {
            int val = i * 2;
            for (int r = 0; r < FILTER_DIM; r++)
                for (int c = 0; c < FILTER_DIM; c++)
                    filters[i][n][r][c] = val++;
        }
    }

    int dprev_dim = dprev[0].size();
    for (int i = 0; i < num_filters; i++)
    {
        int val = i * 2;
        for (int r = 0; r < dprev_dim; r++)
            for (int c = 0; c < dprev_dim; c++)
                dprev[i][r][c] = val++;
    }
}

void test_conv()
{
    int num_filters = 2, num_channels = 3;
    vector<vector<vector<int> > > image(num_channels, vector<vector<int> >(IMAGE_DIM, vector<int>(IMAGE_DIM)));
    //vector<vector<vector<int> > > filters(num_filters, vector<vector<int> >(FILTER_DIM, vector<int>(FILTER_DIM)));
    vector<vector<vector<vector<int> > > > filters(num_filters, vector<vector<vector<int> > >(num_channels, vector<vector<int> >(FILTER_DIM, vector<int>(FILTER_DIM))));
    vector<int> bias(num_filters, 0);
    create_data_forward(image, filters, num_filters, num_channels);
    array3D<int> conv = convolution(image, filters, bias, 1);
    //print_matrix(conv[0]);
}

void test_conv_bp()
{
    int num_filters = 2, num_channels = 3, stride = 1;
    vector<vector<vector<int> > > image(num_channels, vector<vector<int> >(IMAGE_DIM, vector<int>(IMAGE_DIM)));
    vector<vector<vector<vector<int> > > > filters(num_filters, vector<vector<vector<int> > >(num_channels, vector<vector<int> >(FILTER_DIM, vector<int>(FILTER_DIM))));

    int out_dim = floor(((IMAGE_DIM - FILTER_DIM) / stride) + 1);
    vector<vector<vector<int> > > dprev(num_filters, vector<vector<int> >(out_dim, vector<int>(out_dim)));
    
    create_data_back(image, filters, dprev, num_filters);

    int image_dim = image[0].size(); // may have been downsampled
    int dprev_dim = dprev[0].size();
    int out_dim_f = floor(((image_dim - dprev_dim) / stride) + 1); // 6
    vector<vector<vector<int> > > df(num_filters, vector<vector<int> >(out_dim_f, vector<int>(out_dim_f)));

    int out_dim_x = dprev_dim + FILTER_DIM - 1;
    assert(out_dim_x == IMAGE_DIM);
    vector<vector<vector<int> > > dx(num_channels, vector<vector<int> >(out_dim_x, vector<int>(out_dim_x)));

    vector<int> dB(num_filters, 0);
    cout << "back conv image " << endl;
    conv_back(dprev, filters, image, 1, df, dx, dB);
}

// pixels range from 0-255
// 28x28 images = 784 pixels
// each file starts with first 4 bytes int id, 4 bytes int for num images, 4 bytes int for num rows,
// 4 bytes int for num cols, sequence of unsigned bytes for each pixel
uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

// first 4 bytes int id, next 4 bytes int for number of labels, sequence of unsigned bytes for each label
uchar* read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

array3D<uint8_t> convert_to_2d(uchar** images, int image_size, int num_images) {
    int image_dim = (int)sqrt(image_size);
    vector<vector<vector<uint8_t> > > square_images(num_images, vector<vector<uint8_t> >(image_dim, vector<uint8_t>(image_dim)));
    for(int n = 0; n < num_images; n++) {
        vector<vector<uint8_t> > image_vec(image_dim, vector<uint8_t>(image_dim));
        uchar* image = images[n];
        for(int i = 0; i < image_dim; i++) {
            for(int j = 0; j < image_dim; j++) {
                int image_index = (image_dim * i) + j;
                image_vec[i][j] = image[image_index];
            }
        }
        square_images[n] = image_vec;
    }
    return square_images;
}

vector<int> convert_labels(uchar* labels, int num_labels) {
    vector<int> label_vec(num_labels);
    for(int i = 0; i < num_labels; i++) {
        label_vec[i] = labels[i];
    }
    return label_vec;
}

/*
void display_images(array3D<uint8_t> images) {
    //  create grayscale image
    array2D<uint8_t> image = images[0];
    uint8_t image_arr[28*28];
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            int image_index = (28 * i) + j;
            image_arr[image_index] = image[i][j];
        }
    }
    cv::Mat imgGray(28, 28, CV_8UC1, image_arr);

    cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    cv::imshow("test", imgGray);
    cv::waitKey(0);
}
*/

/*
// this works
void display_images_char(uchar** images) {
    uchar* image = images[40000];
    uint8_t image_arr[28*28];
    for(int i = 0; i < 784; i++) {
        image_arr[i] = image[i];
    }
    cv::Mat imgGray(28, 28, CV_8UC1, image_arr);

    cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    cv::imshow("test", imgGray);
    cv::waitKey(0);
}
*/

void test_load_images() {
    int number_of_images = 0;
    int image_size = 0;
    uchar** image_arr = read_mnist_images("/Users/liam_adams/my_repos/csc724_project/data/train-images-idx3-ubyte", number_of_images, image_size);
    cout << "number of images " << number_of_images << endl;
    cout << "image size " << image_size << endl;
    //display_images_char(image_arr); // this works
    array3D<uint8_t> images = convert_to_2d(image_arr, image_size, number_of_images);
    cout << "number of images in vec " << images.size() << endl;
    //display_images(images);
    
    int num_labels = 0;
    uchar* label_arr = read_mnist_labels("/Users/liam_adams/my_repos/csc724_project/data/train-labels-idx1-ubyte", num_labels);
    vector<int> labels = convert_labels(label_arr, num_labels);
    cout << "number of labels " << labels.size() << endl;
}

void test_maxpool() {
    int num_filters = 3, pool_dim = 2, stride = 2, num_channels = 3;
    vector<vector<vector<int> > > image(num_channels, vector<vector<int> >(IMAGE_DIM, vector<int>(IMAGE_DIM)));

    //don't need this var
    vector<vector<vector<vector<int> > > > filters(num_filters, vector<vector<vector<int> > >(num_channels, vector<vector<int> >(FILTER_DIM, vector<int>(FILTER_DIM))));
    
    create_data_forward(image, filters, num_filters, num_channels);
    maxpool_forward(image, pool_dim, stride);
}

void create_data_maxpool_back(array3D<int> &image)
{
    int channels = image.size();
    int image_dim = image[0].size();
    for (int i = 0; i < channels; i++)
    {
        int val = i * 2;
        for (int r = 0; r < image_dim; r++)
            for (int c = 0; c < image_dim; c++)
                image[i][r][c] = val++;
    }
}

void test_maxpool_back() {
    int orig_dim = 20, pool_dim = 2, stride = 2, channels = 3;
    // orig image, say 3 x 5 x 5
    vector<vector<vector<int> > > image(channels, vector<vector<int> >(orig_dim, vector<int>(orig_dim)));
    create_data_maxpool_back(image);
    
    // if pool dim is 2 output of max pool is 3 x 4 x 4
    int new_dim = ((orig_dim - pool_dim) / stride) + 1;
    array3D<int> pooled = maxpool_forward(image, pool_dim, stride);
    cout << "printing pooled" << endl;
    print_matrix(pooled[0]);
    cout << endl;
    
    // dprev is flattened vector, resize to output of max pool, flattened has 48 elements
    vector<int> flattened = flatten(pooled);

    // unflattened is 3 x 3 x 4
    int conv_last_dim = orig_dim;
    int pool_output_dim = ((conv_last_dim - pool_dim) / stride) + 1;
    array3D<int> unflattened = unflatten(flattened, channels, new_dim);
    cout << "printing unflattened" << endl;
    print_matrix(unflattened[0]);
    cout << endl;
    
    // dnext should be 3 x 5 x 5, same as orig image
    array3D<int> dnext = maxpool_backward(unflattened, image, pool_dim, stride);
    cout << "printing orig image" << endl;
    print_matrix(image[0]);
    cout << endl;
    cout << "printing dnext" << endl;
    print_matrix(dnext[0]);
}

int main()
{
    //test_maxpool_back();
    //test_maxpool();
    //test_load_images();
    test_conv_bp();
    //test_conv();
}
