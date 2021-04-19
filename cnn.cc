#include "cnn.h"

Model::Model(int filter_dim, int pool_dim, int num_filters, int pool_stride, int conv_stride, int dense_first_out_dim) : filter_dim(filter_dim), pool_dim(pool_dim), pool_stride(pool_stride), conv_stride(conv_stride), dense_first_out_dim(dense_first_out_dim), maxpool_layer(pool_dim, pool_stride)
{
    int num_channels_first_conv = 1; // num channels in mnist images
    Conv_Layer conv_first(num_filters, filter_dim, num_channels_first_conv, conv_stride, IMAGE_DIM);
    int num_channels_last_conv = num_filters; // output of first conv layer will have num_filters channels
    int input_dim_last_conv = conv_first.get_out_dim();
    Conv_Layer conv_last(num_filters, filter_dim, num_channels_last_conv, conv_stride, input_dim_last_conv);
    vector<Conv_Layer> conv_layers(2);
    conv_layers[0] = conv_first;
    conv_layers[1] = conv_last;

    int conv_last_dim = conv_last.get_out_dim();
    int pool_output_dim = maxpool_layer.get_out_dim(conv_last_dim);
    int flattened_dim = num_filters * pool_output_dim * pool_output_dim;
    Dense_Layer dense_first(flattened_dim, dense_first_out_dim);
    Dense_Layer dense_last(dense_first_out_dim, NUM_LABELS);
    vector<Dense_Layer> dense_layers(2);
    dense_layers[0] = dense_first;
    dense_layers[1] = dense_last;
}

vector<float> Model::forward(array3D<float> &image, vector<uint8_t> &label_one_hot) {
    array3D<float> conv_out_first = conv_layers[0].forward(image);
    relu(conv_out_first);
    array3D<float> conv_out_second = conv_layers[1].forward(conv_out_first);
    relu(conv_out_second);
    array3D<float> pool_out = maxpool_layer.forward(conv_out_second);
    vector<float> flattened = flatten(pool_out);
    vector<float> dense_out_first = dense_layers[0].forward(flattened);
    relu(dense_out_first);
    vector<float> final_out = dense_layers[1].forward(dense_out_first);
    return final_out;
}

void Model::backprop(vector<float> &probs, vector<uint8_t> &labels_one_hot, bool reset_grads)
{
    // one image at a time
    // 2 dense layers dense_first and dense_last
    // vectors are row vectors, transpose are column vectors

    // first step is to subtract probability vector for one hot label vector, this gives dout or dL/dout
    // perfect prediction on all classes will be dout = 0
    vector<float> dout(probs.size());
    for (int i = 0; i < probs.size(); i++)
        dout[i] = probs[i] - labels_one_hot[i];

    // now dot product of dout with input to dense_last transposed, z^T gives dL/dW_last
    // w_last * z + b_last = out, out is probability vector
    // dL/dW_last = dL/dout * dout/dW_last, this is a matrix
    Dense_Layer dense_last = dense_layers[1];
    vector<float> dz = dense_last.backward(dout, reset_grads);

    // dL/db_last = dL/dout * dout/db_last = dL/dout
    // dout/db_last = 1, local gradient of addition operator wrt to val or bias (out = val + bias) is 1
    // sum cols of dL/dout, if dL/dout is vector do nothing

    // now find gradient of last dense layer input or first dense layer output z, dL/dz or dz
    // dout is still gradient of previous layer
    // w_last * z + b_last = out
    // dL/dout * dout/dz, dout/dz = w_last
    // w_last^T * dout = dz

    // now run relu(dz)
    relu(dz);

    // fc is flattened output of max pool or input to dense_first
    // dL/dz * dz/dW_first gives dL/dW_first, this is a matrix
    // dL/db_first is sum of cols of dz

    // find gradient of fc
    // wfirst * fc + b_first = z
    // dL/dz * dz/dfc = dL/dfc
    // w_first^T * dz = dL/dfc
    Dense_Layer dense_first = dense_layers[0];
    vector<float> dfc = dense_first.backward(dz, reset_grads);

    // now reshape dfc (unflatten) to match dimension of max pooling output
    int conv_last_dim = conv_layers[1].get_out_dim();
    int pool_output_dim = maxpool_layer.get_out_dim(conv_last_dim);
    array3D<float> d_pool = unflatten(dfc, num_filters, pool_output_dim);

    // pass this to max_pool.backward() to get dconv2
    array3D<float> dconv2 = maxpool_layer.backward(d_pool);

    // now run relu(dconv2)
    relu(dconv2);

    // run conv.backward(dconv2) = dconv1
    Conv_Layer conv_last = conv_layers[1];
    array3D<float> dconv1 = conv_last.backward(dconv2, reset_grads);

    // run relu(dconv1)
    relu(dconv1);

    // run conv.backward(dconv1) = dimage
    Conv_Layer conv_first = conv_layers[0];
    array3D<float> dimage = conv_first.backward(dconv1, reset_grads);

    // now pass all gradients to adam optimizer
}

vector<Dense_Layer> Model::get_dense_layers() {
    return dense_layers;
}

vector<Conv_Layer> Model::get_conv_layers() {
    return conv_layers;
}

array3D<float> unflatten(vector<float> &vec, int num_filters, int pool_output_dim)
{
    vector<vector<vector<float>>> d_pool(num_filters, vector<vector<float>>(pool_output_dim, vector<float>(pool_output_dim)));
    int row = 0, col = 0, channel = 0, channel_offset = 0;
    for (int i = 0; i < vec.size(); i++)
    {
        channel = floor(i / (pool_output_dim * pool_output_dim));
        channel_offset = channel * pool_output_dim * pool_output_dim;
        row = floor((i - channel_offset) / pool_output_dim);
        col = (i - channel_offset) % pool_output_dim;
        d_pool[channel][row][col] = vec[i];
    }
    return d_pool;
}

vector<float> flatten(array3D<float> &image)
{
    int rows = image[0].size();
    int cols = image[0][0].size();
    int channels = image.size();
    int flattened_dim = channels * rows * cols;
    vector<float> flattened(flattened_dim);
    for (int n = 0; n < channels; n++)
    {
        for (int i = 0; i < rows; i++)
        {
            int k = n * rows * cols; // offset of current square
            for (int j = 0; j < cols; j++)
            {
                int l = (i * cols) + j; // offset within square
                int offset = k + l;
                flattened[offset] = image[n][i][j];
            }
        }
    }
    return flattened;
}

void relu(vector<float> &in)
{
    for (float &p : in)
        if (p < 0)
            p = 0;
}

void relu(array3D<float> &in)
{
    for (vector<vector<float>> &channel : in)
        for (vector<float> &row : channel)
            for (float &val : row)
                if (val < 0)
                    val = 0;
}

void softmax(vector<float> &in)
{
    float sum = 0;
    for (float &p : in)
    {
        exp(p);
        sum += p;
    }
    for (float &p : in)
        p /= sum;
}

float cat_cross_entropy(vector<float> &pred_probs, vector<uint8_t> &true_labels)
{
    int i, tmp = 0;
    float sum = 0;
    for (float p : pred_probs)
    {
        float l = true_labels[i];
        tmp = l * log(p);
        sum += tmp;
        i++;
    }
    return -sum;
}

array2D<float> rotate_180(array2D<float> filter)
{
    reverse(begin(filter), end(filter)); // reverse rows
    for_each(begin(filter), end(filter),
             [](auto &i) { reverse(begin(i), end(i)); }); // reverse columns
    return filter;
}

array2D<float> transpose(array2D<float> &w)
{
    int num_rows = w.size();
    int num_cols = w[0].size();
    vector<vector<float>> t(num_cols, vector<float>(num_rows));
    for (int i = 0; i < num_rows; i++)
    {
        vector<float> row = w[i];
        for (int j = 0; j < num_cols; j++)
        {
            t[j][i] = w[i][j];
        }
    }
    return t;
}

vector<float> dot_product(array2D<float> &w, vector<float> &x)
{
    int rows = w.size();
    int cols = x.size();
    assert(w[0].size() == x.size());
    vector<float> product(rows);
    int tmp = 0;
    for (int i = 0; i < rows; i++)
    {
        float out_value = 0;
        for (int j = 0; j < cols; j++)
        {
            tmp = w[i][j] * x[j];
            out_value += tmp;
        }
        product[i] = out_value;
    }

    return product;
}

void adam(vector<Conv_Layer> &conv_layers, vector<Dense_Layer> &dense_layers, float learning_rate, float beta1, float beta2, int batch_size) {
    // update dF and dB in both conv layers
    array3D<float> conv_first_dF = conv_layers[0].get_dF();
    array3D<float> conv_second_dF = conv_layers[1].get_dF();
    vector<float> conv_first_dB = conv_layers[0].get_dB();
    vector<float> conv_second_dB = conv_layers[1].get_dB();
    int num_filters = conv_first_dF.size();
    int filter_dim = conv_first_dF[0].size();
    float v1 = 0, s1 = 0, v2 = 0, s2 = 0, bv1 = 0, bs1 = 0, bv2 = 0, bs2 = 0;
    float epsilon = .0000001; // to prevent division by zero
    for(int f = 0; f < num_filters; f++) {
        bv1 =  (1 - beta1) * conv_first_dB[f] / batch_size;
        bs1 = (1 - beta2) / pow(conv_first_dB[f] / batch_size, 2);
        conv_first_dB[f] -= learning_rate * bv1/sqrt(bs1 + epsilon);

        bv2 =  (1 - beta1) * conv_second_dB[f] / batch_size;
        bs2 = (1 - beta2) / pow(conv_second_dB[f] / batch_size, 2);
        conv_second_dB[f] -= learning_rate * bv2/sqrt(bs2 + epsilon);
        for(int i = 0; i < filter_dim; i++) {
            for(int j = 0; j < filter_dim; j++) {
                v1 =  (1 - beta1) * conv_first_dF[f][i][j] / batch_size;
                s1 = (1 - beta2) / pow(conv_first_dF[f][i][j] / batch_size, 2);
                conv_first_dF[f][i][j] -= learning_rate * v1/sqrt(s1 + epsilon);

                v2 =  (1 - beta1) * conv_second_dF[f][i][j] / batch_size;
                s2 = (1 - beta2) / pow(conv_second_dF[f][i][j] / batch_size, 2);
                conv_second_dF[f][i][j] -= learning_rate * v1/sqrt(s1 + epsilon);
            }
        }
    }

    // update dW and dB in first dense layer
    array2D<float> dense_first_dW = dense_layers[0].get_dW();
    vector<float> dense_first_dB = dense_layers[0].get_dB(); // size is out_dim
    int dense_first_out_dim = dense_first_dW.size();
    int dense_first_in_dim = dense_first_dW[0].size();
    float v3 = 0, s3 = 0, bv3 = 0, bs3 = 0;
    for(int i = 0; i < dense_first_out_dim; i++) {
        bv3 =  (1 - beta1) * dense_first_dB[i] / batch_size;
        bs3 = (1 - beta2) / pow(dense_first_dB[i] / batch_size, 2);
        dense_first_dB[i] -= learning_rate * bv3/sqrt(bs3 + epsilon);
        for(int j = 0; j < dense_first_in_dim; j++) {
            v3 =  (1 - beta1) * dense_first_dW[i][j] / batch_size;
            s3 = (1 - beta2) / pow(dense_first_dW[i][j] / batch_size, 2);
            dense_first_dW[i][j] -= learning_rate * v3/sqrt(s3 + epsilon);
        }
    }

    // update dW and dB in second dense layer
    array2D<float> dense_second_dW = dense_layers[1].get_dW();
    vector<float> dense_second_dB = dense_layers[1].get_dB(); // size is out_dim
    int out_dim = dense_second_dW.size();
    int in_dim = dense_second_dW[0].size();
    float v4 = 0, s4 = 0, bv4 = 0, bs4 = 0;
    for(int i = 0; i < NUM_LABELS; i++) {
        bv3 =  (1 - beta1) * dense_second_dB[i] / batch_size;
        bs3 = (1 - beta2) / pow(dense_second_dB[i] / batch_size, 2);
        dense_second_dB[i] -= learning_rate * bv3/sqrt(bs3 + epsilon);
        for(int j = 0; j < dense_first_out_dim; j++) {
            v3 =  (1 - beta1) * dense_second_dW[i][j] / batch_size;
            s3 = (1 - beta2) / pow(dense_second_dW[i][j] / batch_size, 2);
            dense_second_dW[i][j] -= learning_rate * v3/sqrt(s3 + epsilon);
        }
    }    
}