#include "cnn.h"
#include "distributed.h"

void read_buf(float* data, Model &cnn, array4D<float> &images, vector<float> &labels) {
    vector<Conv_Layer> &conv_layers = cnn.get_conv_layers();
    array4D<float> &first_conv_filters = conv_layers[0].get_filters();
    vector<float> &first_conv_bias = conv_layers[0].get_bias();
    array4D<float> &second_conv_filters = conv_layers[1].get_filters();
    vector<float> &second_conv_bias = conv_layers[1].get_bias();

    vector<Dense_Layer> &dense_layers = cnn.get_dense_layers();
    array2D<float> &first_dense_weights = dense_layers[0].get_weights();
    vector<float> &first_dense_bias = dense_layers[0].get_bias();
    array2D<float> &second_dense_weights = dense_layers[1].get_weights();
    vector<float> &second_dense_bias = dense_layers[1].get_bias();

    int buf_idx = 0;
    int num_images = images.size();
    for(int n = 0; n < num_images; n++)
        for(int c = 0; c < IMAGE_CHANNELS; c++)
            for(int i = 0; i < IMAGE_DIM; i++)
                for(int j = 0; j < IMAGE_DIM; j++)
                    images[n][c][i][j] = data[buf_idx++];
    assert(buf_idx == num_images * IMAGE_DIM * IMAGE_DIM);

    for(int i = 0; i < num_images; i++)
        labels[i] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images);

    for(int f = 0; f < NUM_FILTERS; f++)
        for(int n = 0; n < IMAGE_CHANNELS; n++)
            for(int i = 0; i < FILTER_DIM; i++)
                for(int j = 0; j < FILTER_DIM; j++)
                        first_conv_filters[f][n][i][j] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images + FIRST_CONV_DF_LEN);

    for(int i = 0; i < NUM_FILTERS; i++)
        first_conv_bias[i] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN);

    for(int f = 0; f < NUM_FILTERS; f++)
        for(int n = 0; n < NUM_FILTERS; n++)
            for(int i = 0; i < FILTER_DIM; i++)
                for(int j = 0; j < FILTER_DIM; j++)
                        second_conv_filters[f][n][i][j] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN);

    for(int i = 0; i < NUM_FILTERS; i++)
        second_conv_bias[i] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN + SECOND_CONV_DB_LEN);

    for(int i = 0; i < DENSE_FIRST_OUT; i++)
        for(int j = 0; j < DENSE_FIRST_IN; j++)
            first_dense_weights[i][j] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN + SECOND_CONV_DB_LEN + FIRST_DENSE_DW_LEN);

    for(int i = 0; i < DENSE_FIRST_OUT; i++)
        first_dense_bias[i] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN + SECOND_CONV_DB_LEN + FIRST_DENSE_DW_LEN + FIRST_DENSE_DB_LEN);
    
    for(int i = 0; i < NUM_LABELS; i++)
        for(int j = 0; j < DENSE_FIRST_OUT; j++)
            second_dense_weights[i][j] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN + SECOND_CONV_DB_LEN + FIRST_DENSE_DW_LEN + FIRST_DENSE_DB_LEN + SECOND_DENSE_DW_LEN);

    for(int i = 0; i < DENSE_FIRST_OUT; i++)
        second_dense_bias[i] = data[buf_idx++];
    assert(buf_idx == (num_images * IMAGE_DIM * IMAGE_DIM) + num_images + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN + SECOND_CONV_DB_LEN + FIRST_DENSE_DW_LEN + FIRST_DENSE_DB_LEN + SECOND_DENSE_DW_LEN + SECOND_DENSE_DB_LEN);
}

vector<float> get_grads(Model &cnn, float batch_loss) {
    vector<float> all_grads;
    all_grads.reserve(FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN + SECOND_CONV_DB_LEN + FIRST_DENSE_DW_LEN + FIRST_DENSE_DB_LEN + SECOND_DENSE_DW_LEN + SECOND_DENSE_DB_LEN + 1);
    
    vector<float>& first_conv_dF = cnn.get_conv_layers()[0].get_flattened_dF();
    vector<float>& first_conv_dB = cnn.get_conv_layers()[0].get_dB();
    vector<float>& second_conv_dF = cnn.get_conv_layers()[1].get_flattened_dF();
    vector<float>& second_conv_dB = cnn.get_conv_layers()[1].get_dB();
    
    vector<float>& first_dense_dW = cnn.get_dense_layers()[0].get_flattened_dW();
    vector<float>& first_dense_dB = cnn.get_dense_layers()[0].get_dB();
    vector<float>& second_dense_dW = cnn.get_dense_layers()[1].get_flattened_dW();
    vector<float>& second_dense_dB = cnn.get_dense_layers()[1].get_dB();
    
    all_grads.insert( all_grads.end(), first_conv_dF.begin(), first_conv_dF.end() );
    all_grads.insert( all_grads.end(), first_conv_dB.begin(), first_conv_dB.end() );
    all_grads.insert( all_grads.end(), second_conv_dF.begin(), second_conv_dF.end() );
    all_grads.insert( all_grads.end(), second_conv_dB.begin(), second_conv_dB.end() );

    all_grads.insert( all_grads.end(), first_dense_dW.begin(), first_dense_dW.end() );
    all_grads.insert( all_grads.end(), first_dense_dB.begin(), first_dense_dB.end() );
    all_grads.insert( all_grads.end(), second_dense_dW.begin(), second_dense_dW.end() );
    all_grads.insert( all_grads.end(), second_dense_dB.begin(), second_dense_dB.end() );
    all_grads.push_back(batch_loss);

    return all_grads;
}

int main(int argc, char const *argv[]) {
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    int batch_size = atoi(argv[1]); // used by adam (not needed for that here) and to get how many images per worker
    int port = atoi(argv[2]);
    cout << "port is " << port << endl;

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
       
    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( port );
       
    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address, 
                                 sizeof(address))<0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    int filter_dim = 5, pool_dim = 2, num_filters = 8, pool_stride = 2, conv_stride = 1, dense_first_out_dim = 128;
    float learning_rate = .01, beta1 = .95, beta2 = .99;
    
    Model cnn(filter_dim, pool_dim, num_filters, pool_stride, conv_stride, dense_first_out_dim, learning_rate, beta1, beta2, batch_size);

    int num_images = batch_size / NUM_WORKERS;
    int images_len = num_images * IMAGE_DIM * IMAGE_DIM;
    
    //
    int buf_len = images_len + NUM_LABELS + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN + SECOND_CONV_DB_LEN + FIRST_DENSE_DW_LEN + FIRST_DENSE_DB_LEN + SECOND_DENSE_DW_LEN + SECOND_DENSE_DB_LEN;
    int buf_size = buf_len * 4;
    float* buffer = new float[buf_len];
    while(true) {
        if (listen(server_fd, 3) < 0)
        {
            perror("listen");
            exit(EXIT_FAILURE);
        }    

        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, 
                       (socklen_t*)&addrlen))<0)
        {
            perror("accept");
            exit(EXIT_FAILURE);
        }
        valread = read( new_socket , buffer, buf_size); // buf_len in bytes
        //printf("%f\n",buffer );
        //print_arr(buffer, 48);
        vector<vector<vector<vector<float> > > > images(num_images, vector<vector<vector<float> > >(IMAGE_CHANNELS, vector<vector<float> >(IMAGE_DIM, vector<float>(IMAGE_DIM, 0))));
        vector<float> labels(num_images, 0);
        read_buf(buffer, cnn, images, labels);
        float image_loss = 0, batch_loss = 0;
        bool reset_grads = true;
        for(int i = 0; i < num_images; i++) {
            array3D<float> image = images[i];
            uint8_t label = labels[i];
            vector<uint8_t> label_one_hot(10, 0);
            label_one_hot[label] = 1;
            vector<float> model_out = cnn.forward(image, label_one_hot);
            image_loss = cat_cross_entropy(model_out, label_one_hot);
            batch_loss += image_loss;
            cnn.backprop(model_out, label_one_hot, reset_grads);
            reset_grads = false;
        }

        vector<float> all_grads = get_grads(cnn, batch_loss);
        float* all_grads_arr = all_grads.data();
        send(new_socket , all_grads_arr , all_grads.size() * 4 , 0 );
        printf("gradients sent\n");
    }
    delete[] buffer;
    return 0;

}