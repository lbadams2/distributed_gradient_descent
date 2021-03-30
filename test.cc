#include <vector>
#include <array>
#include <random>
#include <iostream>
#include <algorithm>

using std::array;
using std::vector;
using std::cout;
using std::endl;
using std::reverse;
using array2D = vector<vector<double> >;
using array3D = vector<vector<vector<double> > >;
#define IMAGE_DIM 3
#define FILTER_DIM 2

void print_matrices(array2D image, array2D filter, array2D conv, int out_dim) {
    cout << "printing image" << endl;
    for(int i = 0; i < IMAGE_DIM; i++) {
        cout << endl;
        for(int j = 0; j < IMAGE_DIM; j++)
            cout << image[i][j] << " ";
    }
    
    cout << "\n\n";
    cout << "printing filter" << endl;
    for(int i = 0; i < FILTER_DIM; i++) {
        cout << endl;
        for(int j = 0; j < FILTER_DIM; j++)
            cout << filter[i][j] << " ";
    }

    cout << "\n\n";
    cout << "printing output" << endl;
    for(int i = 0; i < out_dim; i++) {
        cout << endl;
        for(int j = 0; j < out_dim; j++)
            cout << conv[i][j] << " ";
    }
    cout << "\n\n";
}

// value at top left corner of kernel multiplied by value at bottom right of neighborhood
// size of std::array cannot be set with variable at runtime
array3D convolution(array2D image, array3D filters, int stride)
{
    int num_images = image.size();
    int num_filters = filters.size();
    const int out_dim = floor(((IMAGE_DIM - FILTER_DIM) / stride) + 1); // 6
    vector<vector<vector<double> > > out_conv(num_filters, vector<vector<double> >(out_dim, vector<double>(out_dim)));
    for(int f = 0; f < num_filters; f++) {
        int curr_y = 0, out_y = 0;
        while(curr_y + FILTER_DIM <= IMAGE_DIM) {
            int curr_x = 0, out_x = 0;
            while(curr_x + FILTER_DIM <= IMAGE_DIM) {
                double sum = 0;
                for(int kr = 0; kr < FILTER_DIM; kr++) {
                    for(int kc = 0; kc < FILTER_DIM; kc++) {
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
        print_matrices(image, filters[f], out_conv[f], out_dim);
    }
    return out_conv;
}

array2D rotate_180(array2D filter) {
    reverse(std::begin(filter), std::end(filter)); // reverse rows
    std::for_each(std::begin(filter), std::end(filter),
                [](auto &i) {reverse(std::begin(i), std::end(i));}); // reverse columns
    return filter;
}

array3D conv_back(array3D dprev, array2D image)
{
    int num_images = image.size();
    int num_filters = filters.size();
    int image_dim = image[0].size(); // may have been downsampled
    int dprev_dim = dprev[0].size();
    
    int out_dim_f = floor(((image_dim - dprev_dim) / stride) + 1); // 6
    vector<vector<vector<double> > > df(num_filters, vector<vector<double> >(out_dim, vector<double>(out_dim)));

    out_dim_x = image_dim;
    vector<vector<vector<double> > > dx(num_filters, vector<vector<double> >(out_dim, vector<double>(out_dim)));
    
    for(int f = 0; f < num_filters; f++) {
        int curr_y = 0, out_y = 0;
        
        // calculate dL/dF
        while(curr_y + dprev_dim <= image_dim) {
            int curr_x = 0, out_x = 0;
            while(curr_x + dprev_dim <= image_dim) {
                double sum = 0;
                for(int kr = 0; kr < dprev_dim; kr++) {
                    for(int kc = 0; kc < dprev_dim; kc++) {
                        // dL/dF_ij = dL/dprev_ij * X_ij
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

        // calculate dL/dX
        curr_f = filters[f];
        curr_f = rotate_180(curr_f);
        curr_y = FILTER_DIM, out_y = 0;
        while(curr_y >= 0) {
            curr_x = FILTER_DIM, out_x = 0;
            int y_overlap = (FILTER_DIM - curr_y) + 1;
            int num_dprev_rows = min(y_overlap, dprev_dim);
            while(curr_x >= 0) {
                double sum = 0;
                int x_overlap = (FILTER_DIM - curr_x) + 1;
                int num_dprev_cols = min(x_overlap, dprev_dim);
                for(int kr = 0; kr < num_dprev_rows; kr++) {
                    for(int kc = 0; kc < dprev_dim; kc++) {
                        // dL/dX_ij = dL/dprev_ij * F_ij
                        // dL/dX = conv(rot180(F), dL/dprev), full convolution
                        double prod = dprev[f][kr][kc] * curr_f[curr_y][curr_x];
                        sum += prod;
                    }
                }
                dx[f][out_y][out_x] = sum;
                curr_x += stride;
                out_x++;
            }
            curr_y += stride;
            out_y++;
        }


        //cout << "Printing filter " << f << endl;
        print_matrices(image, filters[f], out_conv[f], out_dim);
    }
    return out_conv;
}

void create_data(array3D &images, array3D &filters, int num_images, int num_filters)
{
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(1, 10);

    for (int i = 0; i < num_images; i++) {
        int val = i * 2;
        for (int r = 0; r < IMAGE_DIM; r++)
            for (int c = 0; c < IMAGE_DIM; c++)
                images[i][r][c] = val++;
    }

    for (int i = 0; i < num_filters; i++) {
        int val = i * 2;
        for (int r = 0; r < FILTER_DIM; r++)
            for (int c = 0; c < FILTER_DIM; c++)
                filters[i][r][c] = val++;
    }
}

void test_conv() {
    int num_images = 3;
    int num_filters = 2;
    vector<vector<vector<double> > > images(num_images, vector<vector<double> >(IMAGE_DIM, vector<double>(IMAGE_DIM)));
    vector<vector<vector<double> > > filters(num_filters, vector<vector<double> >(FILTER_DIM, vector<double>(FILTER_DIM)));
    create_data(images, filters, num_images, num_filters);
    for(int i = 0; i < num_images; i++) {
        cout << "Convolution image " << i << endl;
        convolution(images[i], filters, 1);
    }
}

void test_conv_bp() {

}

int main()
{

    //test_conv();
}