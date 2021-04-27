#include "cnn.h"

Model::Model(int filter_dim, int pool_dim, int num_filters, int pool_stride, int conv_stride, int dense_first_out_dim, float learning_rate, float beta1, float beta2, int batch_size) : filter_dim(filter_dim), pool_dim(pool_dim), num_filters(num_filters), pool_stride(pool_stride), conv_stride(conv_stride), dense_first_out_dim(dense_first_out_dim), maxpool_layer(pool_dim, pool_stride), learning_rate(learning_rate), beta1(beta1), beta2(beta2), batch_size(batch_size)
{
    int num_channels_first_conv = 1; // num channels in mnist images
    Conv_Layer conv_first(num_filters, filter_dim, num_channels_first_conv, conv_stride, IMAGE_DIM);
    int num_channels_last_conv = num_filters; // output of first conv layer will have num_filters channels
    int input_dim_last_conv = conv_first.get_out_dim();
    Conv_Layer conv_last(num_filters, filter_dim, num_channels_last_conv, conv_stride, input_dim_last_conv);
    vector<Conv_Layer> conv_layers(2);
    conv_layers[0] = conv_first;
    conv_layers[1] = conv_last;
    this->conv_layers = conv_layers;

    int conv_last_dim = conv_last.get_out_dim();
    int pool_output_dim = maxpool_layer.get_out_dim(conv_last_dim);
    int flattened_dim = num_filters * pool_output_dim * pool_output_dim;
    Dense_Layer dense_first(flattened_dim, dense_first_out_dim);
    Dense_Layer dense_last(dense_first_out_dim, NUM_LABELS);
    vector<Dense_Layer> dense_layers(2);
    dense_layers[0] = dense_first;
    dense_layers[1] = dense_last;
    this->dense_layers = dense_layers;
    
    // init adam vars
    this->m_df1 = vector<vector<vector<vector<float> > > >(num_filters, vector<vector<vector<float> > >(num_channels_first_conv, vector<vector<float> >(filter_dim, vector<float>(filter_dim, 0))));
    this->m_db1 = vector<float>(num_filters, 0);
    this->v_df1 = vector<vector<vector<vector<float> > > >(num_filters, vector<vector<vector<float> > >(num_channels_first_conv, vector<vector<float> >(filter_dim, vector<float>(filter_dim, 0))));
    this->v_db1 = vector<float>(num_filters, 0);
    
    this->m_df2 = vector<vector<vector<vector<float> > > >(num_filters, vector<vector<vector<float> > >(num_channels_last_conv, vector<vector<float> >(filter_dim, vector<float>(filter_dim, 0))));
    this->m_db2 = vector<float>(num_filters, 0);
    this->v_df2 = vector<vector<vector<vector<float> > > >(num_filters, vector<vector<vector<float> > >(num_channels_last_conv, vector<vector<float> >(filter_dim, vector<float>(filter_dim, 0))));
    this->v_db2 = vector<float>(num_filters, 0);
    
    this->m_dw1 = vector<vector<float> >(dense_first_out_dim, vector<float>(flattened_dim, 0));
    this->m_db3 = vector<float>(dense_first_out_dim, 0);
    this->v_dw1 = vector<vector<float> >(dense_first_out_dim, vector<float>(flattened_dim, 0));
    this->v_db3 = vector<float>(dense_first_out_dim, 0);
    
    this->m_dw2 = vector<vector<float> >(NUM_LABELS, vector<float>(dense_first_out_dim, 0));
    this->m_db4 = vector<float>(NUM_LABELS, 0);
    this->v_dw2 = vector<vector<float> >(NUM_LABELS, vector<float>(dense_first_out_dim, 0));
    this->v_db4 = vector<float>(NUM_LABELS, 0);
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
    softmax(final_out);
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
    Dense_Layer& dense_last = dense_layers.at(1);
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
    Dense_Layer& dense_first = dense_layers.at(0);
    vector<float> dfc = dense_first.backward(dz, reset_grads);

    // now reshape dfc (unflatten) to match dimension of max pooling output
    int conv_last_dim = conv_layers.at(1).get_out_dim();
    int pool_output_dim = maxpool_layer.get_out_dim(conv_last_dim);
    array3D<float> d_pool = unflatten(dfc, num_filters, pool_output_dim);

    // pass this to max_pool.backward() to get dconv2
    array3D<float> dconv2 = maxpool_layer.backward(d_pool);

    // now run relu(dconv2)
    relu(dconv2);

    // run conv.backward(dconv2) = dconv1
    Conv_Layer& conv_last = conv_layers.at(1);
    array3D<float> dconv1 = conv_last.backward(dconv2, reset_grads);

    // run relu(dconv1)
    relu(dconv1);

    // run conv.backward(dconv1) = dimage
    Conv_Layer& conv_first = conv_layers.at(0);
    array3D<float> dimage = conv_first.backward(dconv1, reset_grads);

    // now pass all gradients to adam optimizer
}

void Model::adam() {
    // update dF and dB in both conv layers
    array4D<float> conv_first_dF = conv_layers.at(0).get_dF();
    vector<float> conv_first_dB = conv_layers.at(0).get_dB();

    array4D<float>& conv_first_filters = conv_layers.at(0).get_filters();
    vector<float>& conv_first_filters_flattened = conv_layers.at(0).get_flattened_filter();
    vector<float>& conv_first_bias = conv_layers.at(0).get_bias();

    int num_filters = conv_first_dF.size();
    int num_channels = conv_first_dF[0].size();
    int filter_dim = conv_first_dF[0][0].size();
    float epsilon = .0000001; // to prevent division by zero
    int vec_idx = 0;
    for(int f = 0; f < num_filters; f++) {
        m_db1[f] = beta1*m_db1[f] + (1 - beta1) * (conv_first_dB[f] / batch_size);
        v_db1[f] = beta2*v_db1[f] + (1 - beta2) * pow(conv_first_dB[f] / batch_size, 2);
        conv_first_bias[f] -= learning_rate * m_db1[f]/sqrt(v_db1[f] + epsilon);
        
        for(int n = 0; n < num_channels; n++)
            for(int i = 0; i < filter_dim; i++) {
                for(int j = 0; j < filter_dim; j++) {
                    m_df1[f][n][i][j] = beta1*m_df1[f][n][i][j] + (1 - beta1) * (conv_first_dF[f][n][i][j] / batch_size);
                    v_df1[f][n][i][j] = beta2*v_df1[f][n][i][j] + (1 - beta2) * pow(conv_first_dF[f][n][i][j] / batch_size, 2);
                    conv_first_filters[f][n][i][j] -= learning_rate * m_df1[f][n][i][j]/sqrt(v_df1[f][n][i][j] + epsilon);
                    conv_first_filters_flattened[vec_idx++] = conv_first_filters[f][n][i][j];
                }
            }
    }
    
    array4D<float> conv_second_dF = conv_layers.at(1).get_dF();
    vector<float> conv_second_dB = conv_layers.at(1).get_dB();
    
    array4D<float>& conv_second_filters = conv_layers.at(1).get_filters();
    vector<float>& conv_second_filters_flattened = conv_layers.at(1).get_flattened_filter();
    vector<float>& conv_second_bias = conv_layers.at(1).get_bias();
    num_channels = conv_second_dF[0].size();
    vec_idx = 0;
    for(int f = 0; f < num_filters; f++) {
        m_db2[f] = beta1*m_db2[f] + (1 - beta1) * (conv_second_dB[f] / batch_size);
        v_db2[f] = beta2*v_db2[f] + (1 - beta2) * pow(conv_second_dB[f] / batch_size, 2);
        conv_second_bias[f] -= learning_rate * m_db2[f]/sqrt(v_db2[f] + epsilon);
        
        for(int n = 0; n < num_channels; n++)
            for(int i = 0; i < filter_dim; i++) {
                for(int j = 0; j < filter_dim; j++) {
                    m_df2[f][n][i][j] = beta1*m_df2[f][n][i][j] + (1 - beta1) * (conv_second_dF[f][n][i][j] / batch_size);
                    v_df2[f][n][i][j] = beta2*v_df2[f][n][i][j] + (1 - beta2) * pow(conv_second_dF[f][n][i][j] / batch_size, 2);
                    conv_second_filters[f][n][i][j] -= learning_rate * m_df2[f][n][i][j]/sqrt(v_df2[f][n][i][j] + epsilon);
                    conv_second_filters_flattened[vec_idx++] = conv_second_filters[f][n][i][j];
                }
            }
    }
    

    // update dW and dB in first dense layer
    array2D<float> dense_first_dW = dense_layers.at(0).get_dW();
    vector<float> dense_first_dB = dense_layers.at(0).get_dB(); // size is out_dim

    array2D<float>& dense_first_weights = dense_layers.at(0).get_weights();
    vector<float>& dense_first_weights_flattened = dense_layers.at(0).get_flattened_weights();
    vector<float>& dense_first_bias = dense_layers.at(0).get_bias(); // size is out_dim

    int dense_first_out_dim = dense_first_dW.size();
    int dense_first_in_dim = dense_first_dW[0].size();
    vec_idx = 0;
    for(int i = 0; i < dense_first_out_dim; i++) {
        m_db3[i] = beta1*m_db3[i] + (1 - beta1) * (dense_first_dB[i] / batch_size);
        v_db3[i] = beta2*v_db3[i] + (1 - beta2) * pow(dense_first_dB[i] / batch_size, 2);
        dense_first_bias[i] -= learning_rate * m_db3[i]/sqrt(v_db3[i] + epsilon);
        for(int j = 0; j < dense_first_in_dim; j++) {
            m_dw1[i][j] = beta1*m_dw1[i][j] + (1 - beta1) * (dense_first_dW[i][j] / batch_size);
            v_dw1[i][j] = beta2*v_dw1[i][j] + (1 - beta2) * pow(dense_first_dW[i][j] / batch_size, 2);
            dense_first_weights[i][j] -= learning_rate * m_dw1[i][j]/sqrt(v_dw1[i][j] + epsilon);
            dense_first_weights_flattened[vec_idx++] = dense_first_weights[i][j];
        }
    }

    // update dW and dB in second dense layer
    array2D<float> dense_second_dW = dense_layers.at(1).get_dW();
    vector<float> dense_second_dB = dense_layers.at(1).get_dB(); // size is out_dim

    array2D<float>& dense_second_weights = dense_layers.at(1).get_weights();
    vector<float>& dense_second_weights_flatttened = dense_layers.at(1).get_flattened_weights();
    vector<float>& dense_second_bias = dense_layers.at(1).get_bias(); // size is out_dim
    vec_idx = 0;
    for(int i = 0; i < NUM_LABELS; i++) {
        m_db4[i] = beta1*m_db4[i] + (1 - beta1) * (dense_second_dB[i] / batch_size);
        v_db4[i] = beta2*v_db4[i] + (1 - beta2) * pow(dense_second_dB[i] / batch_size, 2);
        dense_second_bias[i] -= learning_rate * m_db4[i]/sqrt(v_db4[i] + epsilon);
        for(int j = 0; j < dense_first_out_dim; j++) {
            m_dw2[i][j] = beta1*m_dw2[i][j] + (1 - beta1) * (dense_second_dW[i][j] / batch_size);
            v_dw2[i][j] = beta2*v_dw2[i][j] + (1 - beta2) * pow(dense_second_dW[i][j] / batch_size, 2);
            dense_second_weights[i][j] -= learning_rate * m_dw2[i][j]/sqrt(v_dw2[i][j] + epsilon);
            dense_second_weights_flatttened[vec_idx++] = dense_second_weights[i][j];
        }
    }
}

vector<Dense_Layer>& Model::get_dense_layers() {
    return dense_layers;
}

vector<Conv_Layer>& Model::get_conv_layers() {
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
        p = exp(p);
        sum += p;
    }
    for (float &p : in)
        p /= sum;
}

float cat_cross_entropy(vector<float> &pred_probs, vector<uint8_t> &true_labels)
{
    int i = 0;
    float sum = 0, tmp = 0;
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
    float tmp = 0;
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
