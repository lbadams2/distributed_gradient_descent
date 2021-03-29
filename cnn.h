#include <cmath>
#include <array>
#include <vector>
#include <random>
#include <string>

#define IMAGE_DIM 28
#define FILTER_DIM 4

using std::floor;
using std::array;
using std::exp;
using std::log;
using std::normal_distribution;
using std::sqrt;
using std::default_random_engine;
using std::unordered_map;
using std::string;
using array3D = vector<vector<vector<double> > >;
using array2D = vector<vector<double> >;

class Conv_Layer {
public:
    Conv_Layer(int num_filters);
    array3D forward(array2D image, int stride);
    unordered_map<string, unordered_map> backward(vector<double> grad_conv_prev, int filter_ind);
private:
    array3D filters;
    vector<double> bias;
    int num_filters;
    normal_distribution normal_dist;
}

class Dense_Layer {
public:
    Dense_Layer(int in_dim, int out_dim);
    array forward(array in);
private:
    array2D weights;
    vector<double> bias;
    int in_dim;
    int out_dim;
    normal_distribution normal_dist;
}