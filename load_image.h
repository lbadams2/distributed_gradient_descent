#include <string>
#include <iostream>
#include <fstream>
#include <vector>

typedef unsigned char uchar;
using std::string;
using std::ifstream;
using std::ios;
using std::runtime_error;
using std::cout;
using std::endl;
using std::vector;

template<typename T>
using array2D = vector<vector<T> >;

template<typename T>
using array3D = vector<vector<vector<T> > >;