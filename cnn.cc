#include "cnn.h"

void backprop() {
    // one image at a time
    // 2 dense layers dense_first and dense_last
    // vectors are row vectors, transpose are column vectors
    
    // first step is to subtract probability vector for one hot label vector, this gives dout or dL/dout
    // perfect prediction on all classes will be dout = 0
    
    // now dot product of dout with input to dense_last transposed, z^T gives dL/dW_last
    // w_last * z + b_last = out, out is probability vector
    // dL/dW_last = dL/dout * dout/dW_last, this is a matrix
    
    // dL/db_last = dL/dout * dout/db_last = dL/dout
    // dout/db_last = 1, local gradient of addition operator wrt to val or bias (out = val + bias) is 1
    // sum cols of dL/dout, if dL/dout is vector do nothing

    // now find gradient of last dense layer input or first dense layer output z, dL/dz or dz
    // dout is still gradient of previous layer
    // w_last * z + b_last = out
    // dL/dout * dout/dz, dout/dz = w_last
    // w_last^T * dout = dz

    // now run relu(dz)

    // fc is flattened output of max pool or input to dense_first
    // dL/dz * dz/dW_first gives dL/dW_first, this is a matrix
    // dL/db_first is sum of cols of dz

    // find gradient of fc
    // wfirst * fc + b_first = z
    // dL/dz * dz/dfc = dL/dfc
    // w_first^T * dz = dL/dfc
    
    // now reshape dfc (unflatten) to match dimension of max pooling output
    // pass this to max_pool.backward() to get dconv2

    // now run relu(dconv2)

    // run conv.backward(dconv2) = dconv1

    // run relu(dconv1)

    // run conv.backward(dconv1) = dimage

    // now pass all gradients to adam optimizer
}

vector<int> flatten(array2D<int> &image) {
    int rows = image.size();
    int cols = image[0].size();
    int flattened_dim = rows * cols;
    vector<int> flattened(flattened_dim);
    for(int i = 0; i < rows; i++)
        for(int j = 0; j < cols; j++) {
            int k = (i * cols) + j;
            flattened[k] = image[i][j];
        }
    return flattened;
}

void relu(vector<double> &in) {
    for( double &p : in )
        if(p < 0)
            p = 0;
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