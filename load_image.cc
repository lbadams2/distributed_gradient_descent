#include "load_image.h"

// pixels range from 0-255
// 28x28 images = 784 pixels
// each file starts with first 4 bytes int id, 4 bytes int for num images, 4 bytes int for num rows, 
// 4 bytes int for num cols, sequence of unsigned bytes for each pixel
vector<float> read_mnist_images(string full_path, int& number_of_images, int& image_size) {
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
        vector<float> image_vec(number_of_images * image_size);

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
            int vec_offset = i * image_size;
            for(int j = 0; j < image_size; j++) {
                uint8_t tmp = _dataset[i][j];
                float norm = (float)tmp;
                norm = (norm - MNIST_MEAN) / MNIST_STDDEV;
                // could also divide each by 255 to scale [0 - 1]
                image_vec[vec_offset + j] = norm;
            }
        }
        return image_vec;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

// first 4 bytes int id, next 4 bytes int for number of labels, sequence of unsigned bytes for each label
vector<uint8_t> read_mnist_labels(string full_path, int& number_of_labels) {
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

        vector<uint8_t> label_vec(number_of_labels);
        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
            label_vec[i] = _dataset[i];
        }
        return label_vec;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

float image_stddev(array3D<uint8_t> &images, int num_images, int image_dim, int sum, float &mean) {
    int num_pixels = num_images * image_dim * image_dim;
    mean = (float)sum / (float)num_pixels;
    float stddev_sum = 0, tmp = 0;
    for(int n = 0; n < num_images; n++)
        for(int i = 0; i < image_dim; i++)
            for(int j = 0; j < image_dim; j++) {
                tmp = (images[n][i][j] - mean) * (images[n][i][j] - mean);
                stddev_sum += tmp;
            }
    tmp = (1 / (float)num_pixels) * stddev_sum;
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

vector<float> get_training_images() {
    int number_of_images = 0, image_size = 0;
    vector<float> images = read_mnist_images("/Users/liam_adams/my_repos/csc724_project/data/train-images-idx3-ubyte", number_of_images, image_size);
    cout << "number of images " << number_of_images << endl;
    cout << "image size " << image_size << endl;
    int sum = 0, image_dim = 0;
    //float mean = 0;
    //float stddev = image_stddev(images, number_of_images, image_dim, sum, mean);
    //cout << "all images mean " << mean << ", all images stddev " << stddev << endl;
    //array3D<float> images_norm = normalize_images(images, number_of_images, image_dim, mean, stddev);
    return images;
}

vector<uint8_t> get_training_labels() {
    int num_labels = 0;
    vector<uint8_t> labels = read_mnist_labels("/Users/liam_adams/my_repos/csc724_project/data/train-labels-idx1-ubyte", num_labels);
    cout << "number of labels " << labels.size() << endl;
    return labels;
}