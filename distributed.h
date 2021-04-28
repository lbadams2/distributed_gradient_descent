#include <thread>
#include <mutex>
#include <future>
#include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <netdb.h>
#include <unistd.h>
#include <chrono>

using std::thread;
using std::mutex;
using std::lock_guard;
using std::promise;

#define NUM_WORKERS 4
#define FIRST_CONV_DF_LEN 200
#define FIRST_CONV_DB_LEN 8
#define SECOND_CONV_DF_LEN 1600
#define SECOND_CONV_DB_LEN 8
#define FIRST_DENSE_DW_LEN 102400
#define FIRST_DENSE_DB_LEN 128
#define SECOND_DENSE_DW_LEN 1280
#define SECOND_DENSE_DB_LEN 10

/*
int first_dF_len = num_filters * image_channels * filter_dim * filter_dim; // 8 * 1 * 5 * 5 = 200
int first_conv_dB_len = num_filters; // 8
int second_dF_len = num_filters * num_filters * filter_dim * filter_dim; // 8 * 8 * 5 * 5 = 1600
int second_conv_dB_len = num_filters; // 8
int first_dW_len = dense_first_out * dense_first_in; // 128 * 800 = 102400
int first_dense_dB_len = dense_first_out; // 128
int second_dense_dW_len = NUM_LABELS * dense_first_out; // 10 * 128 = 1280
int second_dense_dB_len = NUM_LABELS; // 10
*/