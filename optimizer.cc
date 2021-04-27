#include "load_image.h"
#include "cnn.h"
#include "optimizer.h"

array4D<float> first_dF;
vector<float> first_conv_dB;

array4D<float> second_dF;
vector<float> second_conv_dB;

array2D<float> first_dW;
vector<float> first_dense_dB;

array2D<float> second_dW;
vector<float> second_dense_dB;

mutex grad_mutex;
mutex cout_mutex;

const char* worker1_hostname = "grad_calc_1";
const char* worker2_hostname = "grad_calc_2";
const char* worker3_hostname = "grad_calc_3";
const char* worker4_hostname = "grad_calc_4";

/*
vector<float> flatten_images(array4D<float> &images) {
    int num_images = images.size();
    int images_size = num_images * IMAGE_DIM * IMAGE_DIM;
    vector<float> flattened(images_size);
    int vec_idx = 0;
    for(int n = 0; n < num_images; n++)
        for(int i = 0; i < IMAGE_DIM; i++)
            for(int j = 0; j < IMAGE_DIM; j++)
                flattened[vec_idx++] = images[n][0][i][j];
    return flattened;
}
*/

void init_grads(int num_filters, int image_channels, int filter_dim, int dense_first_in, int dense_first_out) {
    first_dF = vector<vector<vector<vector<float> > > >(num_filters, vector<vector<vector<float> > >(image_channels, vector<vector<float> >(filter_dim, vector<float>(filter_dim, 0))));
    first_conv_dB = vector<float>(num_filters, 0); // 1 bias per filter

    second_dF = vector<vector<vector<vector<float> > > >(num_filters, vector<vector<vector<float> > >(num_filters, vector<vector<float> >(filter_dim, vector<float>(filter_dim, 0))));
    second_conv_dB = vector<float>(num_filters, 0); // 1 bias per filter

    first_dW = vector<vector<float> >(dense_first_out, vector<float>(dense_first_in, 0));
    first_dense_dB = vector<float>(dense_first_out, 0);

    second_dW = vector<vector<float> >(NUM_LABELS, vector<float>(dense_first_out, 0));
    second_dense_dB = vector<float>(NUM_LABELS, 0);
}

array2D<float> create_batches(vector<float> &images, vector<uint8_t> &labels, array2D<float> &worker_images, array2D<float> &worker_labels) {
    int num_worker_images = worker_images.size();
    int pixels_per_worker = worker_images[0].size();
    int image_per_worker = worker_labels[0].size();

    int num_worker_labels = num_worker_images;
    int total_labels = num_worker_labels * image_per_worker;

    int vec_idx = 0;
    int label_idx = 0;
    for(int i = 0; i < num_worker_images; i++) {
        for(int j = 0; j < pixels_per_worker; j++)
            worker_images[i][j] = images[vec_idx++];
        for(int l = 0; l < image_per_worker; l++)
            worker_labels[i][l] = labels[label_idx++];
    }
    assert(label_idx == total_labels);

    return worker_images;
}

vector<float> get_worker_data(vector<float> &flattened_images, vector<float> &labels, Model &cnn) {
    vector<Conv_Layer> &conv_layers = cnn.get_conv_layers();
    vector<float> first_conv_filter = conv_layers[0].get_flattened_filter();
    vector<float> first_conv_bias = conv_layers[0].get_bias();
    vector<float> second_conv_filter = conv_layers[1].get_flattened_filter();
    vector<float> second_conv_bias = conv_layers[1].get_bias();

    vector<Dense_Layer> &dense_layers = cnn.get_dense_layers();
    vector<float> first_dense_weights = dense_layers[0].get_flattened_weights();
    vector<float> first_dense_bias = dense_layers[0].get_bias();
    vector<float> second_dense_weights = dense_layers[1].get_flattened_weights();
    vector<float> second_dense_bias = dense_layers[1].get_bias();

    vector<float> all_data;
    all_data.reserve(flattened_images.size() + labels.size() + first_conv_filter.size() + first_conv_bias.size() + second_conv_filter.size() + second_conv_bias.size() + first_dense_weights.size() + first_dense_bias.size() + second_dense_weights.size() + second_dense_bias.size());
    
    all_data.insert( all_data.end(), flattened_images.begin(), flattened_images.end() );
    all_data.insert( all_data.end(), labels.begin(), labels.end() );
    
    all_data.insert( all_data.end(), first_conv_filter.begin(), first_conv_filter.end() );
    all_data.insert( all_data.end(), first_conv_bias.begin(), first_conv_bias.end() );
    all_data.insert( all_data.end(), second_conv_filter.begin(), second_conv_filter.end() );
    all_data.insert( all_data.end(), second_conv_bias.begin(), second_conv_bias.end() );

    all_data.insert( all_data.end(), first_dense_weights.begin(), first_dense_weights.end() );
    all_data.insert( all_data.end(), first_dense_bias.begin(), first_dense_bias.end() );
    all_data.insert( all_data.end(), second_dense_weights.begin(), second_dense_weights.end() );
    all_data.insert( all_data.end(), second_dense_bias.begin(), second_dense_bias.end() );

    return all_data;
}

string resolve_host(const char* host_name) {
    struct hostent *host_entry;
    const char *IPbuffer;

    host_entry = gethostbyname(host_name);
    IPbuffer = inet_ntoa(*((struct in_addr*)host_entry->h_addr_list[0]));    
    return IPbuffer;
}

int send_vec(float* all_vec_arr, int vec_size, const char* ip_address, int port, int thread_id) {

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
    
    int image_per_worker = batch_size / NUM_WORKERS;
    int pixels_per_worker = image_per_worker * IMAGE_DIM * IMAGE_DIM;
    int num_worker_images = images.size() / pixels_per_worker;
    vector<vector<float> > worker_images(num_worker_images, vector<float>(pixels_per_worker));

    int num_worker_labels = num_worker_images;
    int total_labels = num_worker_labels * image_per_worker;
    assert(total_labels == images.size());
    vector<vector<float> > worker_labels(num_worker_labels, vector<float>(image_per_worker));
    create_batches(images, labels, worker_images, worker_labels);

    // num_filters is number of filters in each conv layer, filter dim is size of filter squares
    int filter_dim = 5, pool_dim = 2, num_filters = 8, pool_stride = 2, conv_stride = 1, dense_first_out_dim = 128;
    float learning_rate = .01, beta1 = .95, beta2 = .99;

    // batch_size just used by adam
    Model cnn(filter_dim, pool_dim, num_filters, pool_stride, conv_stride, dense_first_out_dim, learning_rate, beta1, beta2, batch_size);
    int dense_first_in = cnn.get_dense_layers()[0].get_in_dim();
    init_grads(num_filters, num_channels, filter_dim, dense_first_in, dense_first_out_dim);

    float batch_loss = 0, image_loss = 0;

    string ip_1_str = resolve_host(worker1_hostname);
    string ip_2_str = resolve_host(worker2_hostname);
    string ip_3_str = resolve_host(worker3_hostname);
    string ip_4_str = resolve_host(worker4_hostname);
    const char * ip_1 = ip_1_str.c_str();
    const char * ip_2 = ip_2_str.c_str();
    const char * ip_3 = ip_3_str.c_str();
    const char * ip_4 = ip_4_str.c_str();
    
    int batch_idx = 0;
    for(int n = 0; n < num_worker_images; n + NUM_WORKERS) {
        batch_loss = 0;
        cout << "processing batch " << batch_idx << endl;
        vector<float> worker1_images = worker_images[n];
        vector<float> worker2_images = worker_images[n+1];
        vector<float> worker3_images = worker_images[n+2];
        vector<float> worker4_images = worker_images[n+3];

        vector<float> worker1_labels = worker_labels[n];
        vector<float> worker2_labels = worker_labels[n+1];
        vector<float> worker3_labels = worker_labels[n+2];
        vector<float> worker4_labels = worker_labels[n+3];

        vector<float> worker1_data = get_worker_data(worker1_images, worker1_labels, cnn);
        vector<float> worker2_data = get_worker_data(worker1_images, worker1_labels, cnn);
        vector<float> worker3_data = get_worker_data(worker1_images, worker1_labels, cnn);
        vector<float> worker4_data = get_worker_data(worker1_images, worker1_labels, cnn);

        thread t1(send_vec, worker1_data.data(), worker1_data.size(), ip_1, 8080, 1);
        thread t2(send_vec, worker1_data.data(), worker2_data.size(), ip_2, 8081, 2);
        thread t3(send_vec, worker1_data.data(), worker3_data.size(), ip_3, 8082, 3);
        thread t4(send_vec, worker1_data.data(), worker4_data.size(), ip_4, 8083, 4);
        t1.join();
        t2.join();
        t3.join();
        t4.join();

        cout << "Loss for batch " << n << ": " << batch_loss / batch_size;
        cnn.adam();
    }
}
