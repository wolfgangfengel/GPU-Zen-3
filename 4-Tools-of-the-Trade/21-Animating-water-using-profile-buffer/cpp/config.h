#ifndef GFLUID_CONFIG_H
#define GFLUID_CONFIG_H

// Common headers.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "Eigen/Dense"
#include "config_type.h"

using VectorXI = Eigen::Matrix<int, -1, 1>;

using VectorX = Eigen::Matrix<real, -1, 1>;
using MatrixX = Eigen::Matrix<real, -1, -1>;

using Vector2 = Eigen::Matrix<real, 2, 1>;
using Vector3 = Eigen::Matrix<real, 3, 1>;
using Vector4 = Eigen::Matrix<real, 4, 1>;
using Matrix2 = Eigen::Matrix<real, 2, 2>;
using Matrix3 = Eigen::Matrix<real, 3, 3>;

// Optional flags.
#ifndef WIN32
#define ENABLE_TIMING 1
#if ENABLE_TIMING
#include <sys/time.h>
#endif
#endif

#endif //GFLUID_CONFIG_H
