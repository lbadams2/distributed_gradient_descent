#ifndef shared_h
#define shared_h

#include <vector>
#include <iostream>
#include <assert.h>
#include <chrono>

using std::vector;
using std::cout;
using std::endl;

#define IMAGE_DIM 28
#define NUM_LABELS 10
#define MNIST_MEAN 33.3184f
#define MNIST_STDDEV 73.7704f

template<typename T>
using array2D = vector<vector<T> >;

template<typename T>
using array3D = vector<vector<vector<T> > >;

template<typename T>
using array4D = vector<vector<vector<vector<T> > > >;

template<typename T>
using array5D = vector<vector<vector<vector<vector<T> > > > >;

#endif