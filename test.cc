#include "opencv2/highgui.hpp"
#include <vector>
#include <array>
#include <random>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>

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

#define IMAGE_DIM 3
#define FILTER_DIM 2

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

// value at top left corner of kernel multiplied by value at bottom right of neighborhood
// size of std::array cannot be set with variable at runtime
array3D<int> convolution(array2D<int> image, array3D<int> filters, int stride)
{
    int num_images = image.size();
    int num_filters = filters.size();
    const int out_dim = floor(((IMAGE_DIM - FILTER_DIM) / stride) + 1); // 6
    vector<vector<vector<int> > > out_conv(num_filters, vector<vector<int> >(out_dim, vector<int>(out_dim)));
    for (int f = 0; f < num_filters; f++)
    {
        int curr_y = 0, out_y = 0;
        while (curr_y + FILTER_DIM <= IMAGE_DIM)
        {
            int curr_x = 0, out_x = 0;
            while (curr_x + FILTER_DIM <= IMAGE_DIM)
            {
                double sum = 0;
                for (int kr = 0; kr < FILTER_DIM; kr++)
                {
                    for (int kc = 0; kc < FILTER_DIM; kc++)
                    {
                        double prod = filters[f][kr][kc] * image[curr_y + kr][curr_x + kc];
                        sum += prod;
                    }
                }
                out_conv[f][out_y][out_x] = sum;
                curr_x += stride;
                out_x++;
            }
            curr_y += stride;
            out_y++;
        }
        //cout << "Printing filter " << f << endl;
        print_matrices_conv(image, filters[f], out_conv[f], out_dim);
    }
    return out_conv;
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

void conv_back(array3D<int> dprev, array3D<int> filters, array2D<int> image, int stride, array3D<int> &df, array3D<int> &dx)
{
    int num_images = image.size();
    int num_filters = filters.size();
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
        print_filter(orig_f);
        array2D<int> curr_f = rotate_180(orig_f);
        print_filter(curr_f);
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
        print_matrices_dx(curr_f, dprev[f], dx[f], out_dim_x);
    }
}

void create_data_forward(array3D<int> &images, array3D<int> &filters, int num_images, int num_filters)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(1, 10);

    for (int i = 0; i < num_images; i++)
    {
        int val = i * 2;
        for (int r = 0; r < IMAGE_DIM; r++)
            for (int c = 0; c < IMAGE_DIM; c++)
                images[i][r][c] = val++;
    }

    for (int i = 0; i < num_filters; i++)
    {
        int val = i * 2;
        for (int r = 0; r < FILTER_DIM; r++)
            for (int c = 0; c < FILTER_DIM; c++)
                filters[i][r][c] = val++;
    }
}

void create_data_back(array3D<int> &images, array3D<int> &filters, array3D<int> &dprev, int num_images, int num_filters)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(1, 10);

    for (int i = 0; i < num_images; i++)
    {
        int val = i * 2;
        for (int r = 0; r < IMAGE_DIM; r++)
            for (int c = 0; c < IMAGE_DIM; c++)
                images[i][r][c] = val++;
    }

    for (int i = 0; i < num_filters; i++)
    {
        int val = i * 2;
        for (int r = 0; r < FILTER_DIM; r++)
            for (int c = 0; c < FILTER_DIM; c++)
                filters[i][r][c] = val++;
    }

    for (int i = 0; i < num_filters; i++)
    {
        int val = i * 2;
        for (int r = 0; r < FILTER_DIM; r++)
            for (int c = 0; c < FILTER_DIM; c++)
                dprev[i][r][c] = val++;
    }
}

void test_conv()
{
    int num_images = 3;
    int num_filters = 2;
    vector<vector<vector<int> > > images(num_images, vector<vector<int> >(IMAGE_DIM, vector<int>(IMAGE_DIM)));
    vector<vector<vector<int> > > filters(num_filters, vector<vector<int> >(FILTER_DIM, vector<int>(FILTER_DIM)));
    create_data_forward(images, filters, num_images, num_filters);
    for (int i = 0; i < num_images; i++)
    {
        cout << "Convolution image " << i << endl;
        convolution(images[i], filters, 1);
    }
}

void test_conv_bp()
{
    int num_images = 3;
    int num_filters = 2;
    vector<vector<vector<int> > > images(num_images, vector<vector<int> >(IMAGE_DIM, vector<int>(IMAGE_DIM)));
    vector<vector<vector<int> > > filters(num_filters, vector<vector<int> >(FILTER_DIM, vector<int>(FILTER_DIM)));
    vector<vector<vector<int> > > dprev(num_filters, vector<vector<int> >(FILTER_DIM, vector<int>(FILTER_DIM)));
    create_data_back(images, filters, dprev, num_images, num_filters);

    int image_dim = images[0].size(); // may have been downsampled
    int dprev_dim = dprev[0].size();
    int stride = 1;
    int out_dim_f = floor(((image_dim - dprev_dim) / stride) + 1); // 6
    vector<vector<vector<int> > > df(num_filters, vector<vector<int> >(out_dim_f, vector<int>(out_dim_f)));

    int out_dim_x = dprev_dim + FILTER_DIM - 1;
    vector<vector<vector<int> > > dx(num_filters, vector<vector<int> >(out_dim_x, vector<int>(out_dim_x)));

    for (int i = 0; i < num_images; i++)
    {
        cout << "back conv image " << i << endl;
        conv_back(dprev, filters, images[i], 1, df, dx);
    }
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
    display_images(images);
    
    int num_labels = 0;
    uchar* label_arr = read_mnist_labels("/Users/liam_adams/my_repos/csc724_project/data/train-labels-idx1-ubyte", num_labels);
    vector<int> labels = convert_labels(label_arr, num_labels);
    cout << "number of labels " << labels.size() << endl;
}

int main()
{
    test_load_images();
    //test_conv_bp();
    //test_conv();
}