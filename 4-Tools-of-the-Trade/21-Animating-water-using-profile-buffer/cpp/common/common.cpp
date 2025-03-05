#include <iostream>
#include <fstream>
#include "common/common.h"
#include "Eigen/SVD"
#include <iomanip>

const real ToReal(const double v) {
    return static_cast<real>(v);
}

const std::string GreenHead() {
  return "\x1b[6;30;92m";
}

const std::string RedHead() {
  return "\x1b[6;30;91m";
}

const std::string CyanHead() {
  return "\x1b[6;30;96m";
}

const std::string GreenTail() {
  return "\x1b[0m";
}

const std::string RedTail() {
  return "\x1b[0m";
}

const std::string CyanTail() {
  return "\x1b[0m";
}

// Timing.
#ifndef WIN32
static struct timeval t_begin, t_end;
#endif

void Tic() {
#if ENABLE_TIMING
    gettimeofday(&t_begin, nullptr);
#endif
}

void Toc(const std::string& message) {
#if ENABLE_TIMING
    gettimeofday(&t_end, nullptr);
    const real t_interval = (t_end.tv_sec - t_begin.tv_sec) + (t_end.tv_usec - t_begin.tv_usec) / 1e6;
    std::cout << CyanHead() << "[Timing] " << message << ": " << t_interval << "s"
              << CyanTail() << std::endl;
#endif

}