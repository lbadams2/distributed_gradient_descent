#include "load_image.h"

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

array3D<uint8_t> convert_to_2d(uchar** images, int image_size, int num_images, int &image_dim, int &sum) {
    int image_dim = (int)sqrt(image_size);
    vector<vector<vector<uint8_t> > > square_images(num_images, vector<vector<uint8_t> >(image_dim, vector<uint8_t>(image_dim)));
    for(int n = 0; n < num_images; n++) {
        vector<vector<uint8_t> > image_vec(image_dim, vector<uint8_t>(image_dim));
        uchar* image = images[n];
        for(int i = 0; i < image_dim; i++) {
            for(int j = 0; j < image_dim; j++) {
                int image_index = (image_dim * i) + j;
                image_vec[i][j] = image[image_index];
                //image_vec[i][j] /= 255; // make everything [0, 1]
                sum += image[image_index];
            }
        }
        square_images[n] = image_vec;
    }
    return square_images;
}

vector<uint8_t> convert_labels(uchar* labels, int num_labels) {
    vector<uint8_t> label_vec(num_labels);
    for(int i = 0; i < num_labels; i++) {
        label_vec[i] = labels[i];
    }
    return label_vec;
}

float image_stddev(array3D<uint8_t> &images, int num_images, int image_dim, int sum, float &mean) {
    int num_pixels = num_images * image_dim * image_dim;
    mean = sum / num_pixels;
    float stddev_sum = 0, tmp = 0;
    for(int n = 0; n < num_images; n++)
        for(int i = 0; i < image_dim; i++)
            for(int j = 0; j < image_dim; j++) {
                tmp = (images[n][i][j] - mean) * (images[n][i][j] - mean);
                stddev_sum += tmp;
            }
    tmp = (1 / num_pixels) * stddev_sum;
    float stddev = sqrt(tmp);
    return stddev;
}

array3D<float> normalize_images(array3D<uint8_t> &images, int num_images, int image_dim, float mean, float stddev) {
    vector<vector<vector<float> > > norm_images(num_images, vector<vector<float> >(image_dim, vector<float>(image_dim)));
    float tmp = 0;
    for(int n = 0; n < num_images; n++)
        for(int i = 0; i < image_dim; i++)
            for(int j = 0; j < image_dim; j++) {
                tmp = images[n][i][j] - mean;
                images[n][i][j] = tmp / stddev;
            }
    return norm_images;
}

array3D<float> get_training_images() {
    int number_of_images = 0, image_size = 0;
    uchar** image_arr = read_mnist_images("/Users/liam_adams/my_repos/csc724_project/data/train-images-idx3-ubyte", number_of_images, image_size);
    cout << "number of images " << number_of_images << endl;
    cout << "image size " << image_size << endl;
    int sum = 0, image_dim = 0;
    float mean = 0;
    array3D<uint8_t> images = convert_to_2d(image_arr, image_size, number_of_images, image_dim, sum);
    float stddev = image_stddev(images, number_of_images, image_dim, sum, mean);
    array3D<float> images_norm = normalize_images(images, number_of_images, image_dim, mean, stddev);
    return images_norm;
}

vector<uint8_t> get_training_labels() {
    int num_labels = 0;
    uchar* label_arr = read_mnist_labels("/Users/liam_adams/my_repos/csc724_project/data/train-labels-idx1-ubyte", num_labels);
    vector<uint8_t> labels = convert_labels(label_arr, num_labels);
    cout << "number of labels " << labels.size() << endl;
    return labels;
}

/*
int main() {
    int number_of_images = 0, image_size = 0;
    uchar** image_arr = read_mnist_images("/Users/liam_adams/my_repos/csc724_project/data/train-images-idx3-ubyte", number_of_images, image_size);
    cout << "number of images " << number_of_images << endl;
    cout << "image size " << image_size << endl;
    int sum = 0, image_dim = 0;
    float mean = 0;
    array3D<uint8_t> images = convert_to_2d(image_arr, image_size, number_of_images, image_dim, sum);
    float stddev = image_stddev(images, number_of_images, image_dim, sum, mean);
    array3D<float> images_norm = normalize_images(images, number_of_images, image_dim, mean, stddev);
    cout << "number of images in vec " << images.size() << endl;
    
    int num_labels = 0;
    uchar* label_arr = read_mnist_labels("/Users/liam_adams/my_repos/csc724_project/data/train-labels-idx1-ubyte", num_labels);
    vector<uint8_t> labels = convert_labels(label_arr, num_labels);
    cout << "number of labels " << labels.size() << endl;
    delete image_arr;
    delete label_arr;
}
*/