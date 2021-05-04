#ifndef shared_h
#define shared_h

#include <vector>
#include <iostream>
#include <assert.h>
#include <fstream>

using std::vector;
using std::cout;
using std::endl;

#define IMAGE_DIM 28
#define NUM_LABELS 10
#define MNIST_MEAN 33.3184f
#define MNIST_STDDEV 73.7704f

#define NUM_FILTERS 8
#define IMAGE_CHANNELS 1
#define FILTER_DIM 5
#define DENSE_FIRST_OUT 128
#define DENSE_FIRST_IN 800

template<typename T>
using array2D = vector<vector<T> >;

template<typename T>
using array3D = vector<vector<vector<T> > >;

template<typename T>
using array4D = vector<vector<vector<vector<T> > > >;

template<typename T>
using array5D = vector<vector<vector<vector<vector<T> > > > >;

#endif