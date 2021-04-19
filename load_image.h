#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "shared.h"

typedef unsigned char uchar;
using std::string;
using std::ifstream;
using std::ios;
using std::runtime_error;
using std::cout;
using std::endl;
using std::sqrt;

array3D<float> get_training_images();
vector<uint8_t> get_training_labels();