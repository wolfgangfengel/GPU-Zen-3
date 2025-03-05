#ifndef GFLUID_COMMON_H
#define GFLUID_COMMON_H

#include "config.h"

const real ToReal(const double v);

// Colorful print.
const std::string GreenHead();
const std::string RedHead();
const std::string CyanHead();
const std::string GreenTail();
const std::string RedTail();
const std::string CyanTail();

// Timing.
void Tic();
void Toc(const std::string& message);
#endif //GFLUID_COMMON_H
