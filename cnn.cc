#include "cnn.h"

void Model::backprop(vector<double> probs, vector<int> labels_one_hot) {
    // one image at a time
    // 2 dense layers dense_first and dense_last
    // vectors are row vectors, transpose are column vectors
    
    // first step is to subtract probability vector for one hot label vector, this gives dout or dL/dout
    // perfect prediction on all classes will be dout = 0
    vector<double> dout(probs.size());
    for(int i = 0; i < probs.size(); i++)
        dout[i] = probs[i] - labels_one_hot[i];
    
    // now dot product of dout with input to dense_last transposed, z^T gives dL/dW_last
    // w_last * z + b_last = out, out is probability vector
    // dL/dW_last = dL/dout * dout/dW_last, this is a matrix
    Dense_Layer dense_last = dense_layers[1];
    vector<double> dz = dense_last.backward(dout);
    
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
    vector<double> dfc = dense_first.backward(dz);
    
    // now reshape dfc (unflatten) to match dimension of max pooling output
    vector<vector<double>> d_pool(pool_dim, vector<double>(pool_dim));
    int row = 0, col = 0;
    for(int i = 0; i < dfc.size(); i++) {
        row = floor(i / pool_dim);
        col = i % pool_dim;
        d_pool[row][col];
    }

    // pass this to max_pool.backward() to get dconv2
    array2D<double> dconv2 = maxpool_layer.backward(d_pool);

    // now run relu(dconv2)
    relu(dconv2);
    
    // run conv.backward(dconv2) = dconv1


    // run relu(dconv1)

    // run conv.backward(dconv1) = dimage

    // now pass all gradients to adam optimizer
}

vector<double> flatten(array3D<double> &image) {
    int rows = image[0].size();
    int cols = image[0][0].size();
    int channels = image.size();
    int flattened_dim = channels * rows * cols;
    vector<double> flattened(flattened_dim);
    for(int n = 0; n < channels; n++) {
        for(int i = 0; i < rows; i++) {
            int k = n * rows * cols; // offset of current square
            for(int j = 0; j < cols; j++) {
                int l = (i * cols) + j; // offset within square
                int offset = k + l;
                flattened[offset] = image[n][i][j];
            }
        }
    }
    return flattened;
}

void relu(vector<double> &in) {
    for( double &p : in )
        if(p < 0)
            p = 0;
}

void relu(array2D<double> &in) {
    for( vector<double> &row : in )
        for(double &val : row)
            if(val < 0)
                val = 0;
}

void softmax(vector<double> &in) {
    double sum = 0;
    for( double &p : in ) {
        exp(p);
        sum += p;
    }
    for( double &p : in )
        p /= sum;
}

double cat_cross_entropy(vector<double> &pred_probs, vector<double> &true_labels) {
    int i, tmp = 0;
    double sum = 0;
    for( double p : pred_probs) {
        double l = true_labels[i];
        tmp = l * log(p);
        sum += tmp;
        i++;
    }
    return -sum;
}

array2D<int> rotate_180(array2D<int> filter)
{
    reverse(begin(filter), end(filter)); // reverse rows
    for_each(begin(filter), end(filter),
                  [](auto &i) { reverse(begin(i), end(i)); }); // reverse columns
    return filter;
}

array2D<double> transpose(array2D<double> w) {
    int num_rows = w.size();
    int num_cols = w[0].size();
    vector<vector<double> > t(num_cols, vector<double>(num_rows));
    for(int i = 0; i < num_rows; i++) {
        vector<double> row = w[i];
        for(int j = 0; j < num_cols; j++) {
            t[j][i] = w[i][j];
        }
    }
    return t;
}

vector<double> dot_product(array2D<double> &w, vector<double> &x) {
    int rows = w.size();
    int cols = x.size();
    vector<double> product(rows);
    int tmp = 0;
    for (int i = 0; i < rows; i++)
    {
        double out_value = 0;
        for (int j = 0; j < cols; j++)
        {
            tmp = w[i][j] * x[j];
            out_value += tmp;
        }
        product[i] = out_value;
    }

    return product;
}