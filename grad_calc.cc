#include "cnn.h"
#include "distributed.h"

void print_cnn(Model &cnn) {
    vector<Conv_Layer> &conv_layers = cnn.get_conv_layers();
    Conv_Layer& second_conv_layer = conv_layers.at(1);
    
    array4D<float>& second_filters = second_conv_layer.get_filters();
    cout << "printing some values from second conv filters" << endl;
    cout << second_filters[0][1][0][1] << " " << second_filters[1][2][0][1] << " " << second_filters[2][3][2][1] << " " << second_filters[3][4][3][1] << endl;
    cout << "\n\n";
    
    vector<float>& second_conv_bias = second_conv_layer.get_bias();
    cout << "printing some values from second conv bias" << endl;
    cout << second_conv_bias[0] << " " << second_conv_bias[1] << " " << second_conv_bias[2] << " " << second_conv_bias[3] << endl;
    cout << "\n\n";

    vector<Dense_Layer> &dense_layers = cnn.get_dense_layers();
    Dense_Layer& first_dense_layer = dense_layers.at(0);

    array2D<float>& first_weights = first_dense_layer.get_weights();
    cout << "printing some values from first dense weights" << endl;
    cout << first_weights[0][10] << " " << first_weights[5][20] << " " << first_weights[18][30] << " " << first_weights[31][50] << endl;
    cout << "\n\n";

    vector<float>& first_dense_bias = first_dense_layer.get_bias();
    cout << "printing some values from first dense bias" << endl;
    cout << first_dense_bias[0] << " " << first_dense_bias[10] << " " << first_dense_bias[20] << " " << first_dense_bias[80] << endl;
    cout << "\n\n";
}

void print_grads(float* grads) {
    cout << "printing some values sent to optimizer, same values it printed" << endl;
    cout << grads[0] << " " << grads[20] << " " << grads[40] << " " << grads[80] << endl;
    cout << "\n\n";
}

void read_buf(vector<float> &data, Model &cnn, array4D<float> &images, vector<float> &labels) {
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

    for(int i = 0; i < NUM_LABELS; i++)
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

void print_buf(vector<float> &buf) {
    cout << "printing values received by optimizer, same values it printed before sending" << endl;
    cout << buf[0] << " " << buf[100] << " " << buf[200] << " " << buf[1000] << endl;
    cout << "\n\n";
}

void fill_buf(float* partial_buf, vector<float> &all_buf, int start_idx, int partial_buf_bytes) {
    int partial_buf_len = partial_buf_bytes / 4;
    for(int i = 0; i < partial_buf_len; i++)
        all_buf.push_back(partial_buf[i]);   
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
    
    int buf_len = images_len + num_images + FIRST_CONV_DF_LEN + FIRST_CONV_DB_LEN + SECOND_CONV_DF_LEN + SECOND_CONV_DB_LEN + FIRST_DENSE_DW_LEN + FIRST_DENSE_DB_LEN + SECOND_DENSE_DW_LEN + SECOND_DENSE_DB_LEN;
    //cout << "buf len for data from optimizer " << buf_len << endl;
    long buf_size = buf_len * 4;
    float* buffer = new float[buf_len];
    int batch_idx = 0;
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    long duration = 0;
    std::ofstream myfile;
    string file_name("runtime_metrics/grad_calc" + std::to_string(port) + "_" + std::to_string(batch_size) + ".txt");
    myfile.open(file_name);
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
        vector<float> all_buffer;
        all_buffer.reserve(buf_len);
        //cout << "buf size read from optimizer " << buf_size << endl;
        int bytes_read = 0;
        while(bytes_read < buf_size) {
            valread = read( new_socket , buffer, buf_size); // buf_len in bytes
            int start_idx = bytes_read / 4;
            bytes_read += valread;
            fill_buf(buffer, all_buffer, start_idx, valread);
        }
        /*
        cout << "total bytes read " << bytes_read << ", " << "number of float elements " << bytes_read / 4 << endl;
        cout << "value of first dense weight " << all_buffer[6272 + 8 + 200 + 8 + 1600 + 8] << endl;
        cout << "value of 80th dense weight " << all_buffer[6272 + 8 + 200 + 8 + 1600 + 8 + 80] << endl;
        */
        //print_buf(all_buffer);

        vector<vector<vector<vector<float> > > > images(num_images, vector<vector<vector<float> > >(IMAGE_CHANNELS, vector<vector<float> >(IMAGE_DIM, vector<float>(IMAGE_DIM, 0))));
        vector<float> labels(num_images, 0);
        read_buf(all_buffer, cnn, images, labels);
        print_cnn(cnn);

        float image_loss = 0, batch_loss = 0;
        bool reset_grads = true;
        start_time = std::chrono::high_resolution_clock::now();
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
        end_time = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto ms_int_count = ms_int.count();
        cout << "time for batch " << batch_idx << ": " << ms_int_count << "ms\n\n\n";
        myfile << ms_int_count << "\n";
        
        //if(batch_idx >= 298)
        //    myfile.flush();
        //if(batch_idx >= 148)
        //    myfile.flush();
        //if(batch_idx >= 72)
        //    myfile.flush();
        
        vector<float> all_grads = get_grads(cnn, batch_loss);
        float* all_grads_arr = all_grads.data();
        //print_grads(all_grads_arr);
        send(new_socket , all_grads_arr , all_grads.size() * 4 , 0 );
        printf("gradients sent\n");
        batch_idx++;
    }
    myfile.close();
    return 0;

}
