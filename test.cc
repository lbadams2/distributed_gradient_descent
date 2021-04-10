#include <vector>
#include <array>
#include <random>
#include <iostream>
#include <algorithm>

using std::array;
using std::cout;
using std::endl;
using std::reverse;
using std::vector;
using std::min;
using array2D = vector<vector<double> >;
using array3D = vector<vector<vector<double> > >;
#define IMAGE_DIM 3
#define FILTER_DIM 2

void print_matrices_conv(array2D image, array2D filter, array2D conv, int out_dim)
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

void print_matrices_df(array2D image, array2D dprev, array2D df, int out_dim)
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

void print_matrices_dx(array2D filter, array2D dprev, array2D dx, int out_dim)
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
array3D convolution(array2D image, array3D filters, int stride)
{
    int num_images = image.size();
    int num_filters = filters.size();
    const int out_dim = floor(((IMAGE_DIM - FILTER_DIM) / stride) + 1); // 6
    vector<vector<vector<double> > > out_conv(num_filters, vector<vector<double> >(out_dim, vector<double>(out_dim)));
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

array2D rotate_180(array2D filter)
{
    reverse(std::begin(filter), std::end(filter)); // reverse rows
    std::for_each(std::begin(filter), std::end(filter),
                  [](auto &i) { reverse(std::begin(i), std::end(i)); }); // reverse columns
    return filter;
}

void print_filter(array2D filter) {
    cout << "printing filter" << endl;
    for (int i = 0; i < FILTER_DIM; i++)
    {
        cout << endl;
        for (int j = 0; j < FILTER_DIM; j++)
            cout << filter[i][j] << " ";
    }
    cout << endl;
}

void conv_back(array3D dprev, array3D filters, array2D image, int stride, array3D &df, array3D &dx)
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
        array2D orig_f = filters[f];
        print_filter(orig_f);
        array2D curr_f = rotate_180(orig_f);
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

void create_data_forward(array3D &images, array3D &filters, int num_images, int num_filters)
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

void create_data_back(array3D &images, array3D &filters, array3D &dprev, int num_images, int num_filters)
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
    vector<vector<vector<double> > > images(num_images, vector<vector<double> >(IMAGE_DIM, vector<double>(IMAGE_DIM)));
    vector<vector<vector<double> > > filters(num_filters, vector<vector<double> >(FILTER_DIM, vector<double>(FILTER_DIM)));
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
    vector<vector<vector<double> > > images(num_images, vector<vector<double> >(IMAGE_DIM, vector<double>(IMAGE_DIM)));
    vector<vector<vector<double> > > filters(num_filters, vector<vector<double> >(FILTER_DIM, vector<double>(FILTER_DIM)));
    vector<vector<vector<double> > > dprev(num_filters, vector<vector<double> >(FILTER_DIM, vector<double>(FILTER_DIM)));
    create_data_back(images, filters, dprev, num_images, num_filters);

    int image_dim = images[0].size(); // may have been downsampled
    int dprev_dim = dprev[0].size();
    int stride = 1;
    int out_dim_f = floor(((image_dim - dprev_dim) / stride) + 1); // 6
    vector<vector<vector<double> > > df(num_filters, vector<vector<double> >(out_dim_f, vector<double>(out_dim_f)));

    int out_dim_x = dprev_dim + FILTER_DIM - 1;
    vector<vector<vector<double> > > dx(num_filters, vector<vector<double> >(out_dim_x, vector<double>(out_dim_x)));

    for (int i = 0; i < num_images; i++)
    {
        cout << "back conv image " << i << endl;
        conv_back(dprev, filters, images[i], 1, df, dx);
    }
}

int main()
{
    test_conv_bp();
    //test_conv();
}