#include "load_image.h"
#include "cnn.h"

// num_batches x batch_size x num_channels x image_dim x image_dim
// probably fastest to pass 1D vector here and get rid of create2D in load images
void create_batches(array3D<float> &images, vector<uint8_t> &labels, array5D<float> &image_batches, array2D<uint8_t> &label_batches) {
    int num_batches = image_batches.size();
    int batch_size = image_batches[0].size();
    for(int n = 0; n < num_batches; n++) {
        for(int b = 0; b < batch_size; b++) {
            int orig_index = (n * b) + b;
            label_batches[n][b] = labels[orig_index];
            for(int i = 0; i < IMAGE_DIM; i++)
                for(int j = 0; j < IMAGE_DIM; j++) {
                    image_batches[n][b][0][i][j] = images[orig_index][i][j];
                }
        }
    }
}

int main() {
    array3D<float> images = get_training_images();
    vector<uint8_t> labels = get_training_labels();

    int batch_size = 25;
    int num_channels = 1; // grayscale
    int num_images = images.size();
    int num_labels = labels.size();
    assert(num_images == num_labels);
    int num_batches = (int)(num_images / batch_size);
    vector<vector<vector<vector<vector<float> > > > > image_batches(num_batches, vector<vector<vector<vector<float> > > >(batch_size, vector<vector<vector<float> > >(num_channels, vector<vector<float> >(IMAGE_DIM, vector<float>(IMAGE_DIM)))));
    //vector<vector<vector<uint8_t> > > label_batches(num_batches, vector<vector<uint8_t> >(batch_size, vector<uint8_t>(IMAGE_DIM)));
    vector<vector<uint8_t>> label_batches(num_batches, vector<uint8_t>(batch_size));
    create_batches(images, labels, image_batches, label_batches);

    // num_filters is number of filters in each conv layer, filter dim is size of filter squares
    int filter_dim = 5, pool_dim = 2, num_filters = 8, pool_stride = 2, conv_stride = 1, dense_first_out_dim = 128;
    Model cnn(filter_dim, pool_dim, num_filters, pool_stride, conv_stride, dense_first_out_dim);    
    
    vector<Dense_Layer> dense_layers = cnn.get_dense_layers();
    vector<Conv_Layer> conv_layers = cnn.get_conv_layers();
    
    float batch_loss = 0, image_loss = 0;
    float learning_rate = .01, beta1 = .95, beta2 = .99;
    bool reset_grads = true;
    for(int n = 0; n < num_batches; n++) {
        batch_loss = 0;        
        reset_grads = true;
        for(int b = 0; b < batch_size; b++) {
            array3D<float> image = image_batches[n][b];
            uint8_t label = label_batches[n][b];
            vector<uint8_t> label_one_hot(10, 0);
            label_one_hot[label] = 1;
            vector<float> model_out = cnn.forward(image, label_one_hot);
            image_loss = cat_cross_entropy(model_out, label_one_hot);
            batch_loss += image_loss;
            cnn.backprop(model_out, label_one_hot, reset_grads);
            reset_grads = false;
        }
        cout << "Loss for batch " << n << ": " << batch_loss / batch_size;
        adam(conv_layers, dense_layers, learning_rate, beta1, beta2, batch_size);
    }
}