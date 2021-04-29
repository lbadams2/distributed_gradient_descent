#include "load_image.h"
#include "cnn.h"

void create_batches(vector<float> &images, vector<uint8_t> &labels, array5D<float> &image_batches, array2D<uint8_t> &label_batches) {
    int num_batches = image_batches.size();
    int batch_size = image_batches[0].size();
    int image_idx = 0, pixel_idx = 0, vec_idx = 0, pixel_offset = 0;
    for(int n = 0; n < num_batches; n++) {
        for(int b = 0; b < batch_size; b++) {
            image_idx = (n * batch_size) + b;
            label_batches[n][b] = labels[image_idx];
            pixel_offset = image_idx * 784;
            for(int i = 0; i < IMAGE_DIM; i++)
                for(int j = 0; j < IMAGE_DIM; j++) {
                    pixel_idx = (i * IMAGE_DIM) + j;
                    vec_idx = pixel_offset + pixel_idx;
                    image_batches[n][b][0][i][j] = images[vec_idx];
                }
        }
    }
}

int main(int argc, char const *argv[]) {
    vector<float> images = get_training_images();
    vector<uint8_t> labels = get_training_labels();

    int batch_size = atoi(argv[1]);
    int num_channels = 1; // grayscale
    int image_size = IMAGE_DIM * IMAGE_DIM;
    int num_images = images.size() / image_size;
    int num_labels = labels.size();
    assert(num_images == num_labels);
    int num_batches = (int)(num_images / batch_size);
    vector<vector<vector<vector<vector<float> > > > > image_batches(num_batches, vector<vector<vector<vector<float> > > >(batch_size, vector<vector<vector<float> > >(num_channels, vector<vector<float> >(IMAGE_DIM, vector<float>(IMAGE_DIM)))));
    //vector<vector<vector<uint8_t> > > label_batches(num_batches, vector<vector<uint8_t> >(batch_size, vector<uint8_t>(IMAGE_DIM)));
    vector<vector<uint8_t>> label_batches(num_batches, vector<uint8_t>(batch_size));
    //cout << "creating batches" << endl;
    create_batches(images, labels, image_batches, label_batches);
    //cout << "done creating batches" << endl;
    // num_filters is number of filters in each conv layer, filter dim is size of filter squares
    int filter_dim = 5, pool_dim = 2, num_filters = 8, pool_stride = 2, conv_stride = 1, dense_first_out_dim = 128;
    float learning_rate = .01, beta1 = .95, beta2 = .99;
    Model cnn(filter_dim, pool_dim, num_filters, pool_stride, conv_stride, dense_first_out_dim, learning_rate, beta1, beta2, batch_size);
    
    float batch_loss = 0, image_loss = 0;
    bool reset_grads = true;
    std::chrono::high_resolution_clock::time_point batch_start_time, batch_end_time, total_start_time, total_end_time;
    long duration = 0;
    int num_demo_images = 4800;
    int num_demo_images_idx = 0;
    total_start_time = std::chrono::high_resolution_clock::now();
    for(int n = 0; n < num_batches; n++) {
        batch_start_time = std::chrono::high_resolution_clock::now();
        batch_loss = 0;
        reset_grads = true;
        cout << "processing batch " << n << endl;
        for(int b = 0; b < batch_size; b++) {
            //cout << "processing image " << b << endl;
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
        cout << "Loss for batch " << n << ": " << batch_loss / batch_size << endl;
        cnn.adam();
        batch_end_time = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end_time - batch_start_time);
        cout << "time for batch " << n << ": " << ms_int.count() << "ms\n\n\n";
        num_demo_images_idx += batch_size;
        if(num_demo_images_idx >= num_demo_images)
            break;
    }
    total_end_time = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
    cout << "time for " << num_demo_images << " images " << ms_int.count() << "ms\n\n\n";
    // takes 190 seconds for 4800 images and batch size of 32
}
