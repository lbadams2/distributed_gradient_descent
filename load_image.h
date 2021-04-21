#include <string>
#include <fstream>
#include <cmath>
#include "shared.h"

typedef unsigned char uchar;
using std::string;
using std::ifstream;
using std::ios;
using std::runtime_error;
using std::sqrt;

vector<float> get_training_images();
vector<uint8_t> get_training_labels();