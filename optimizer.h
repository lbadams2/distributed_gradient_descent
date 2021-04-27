#include <thread>
#include <mutex>
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

#define NUM_WORKERS 4