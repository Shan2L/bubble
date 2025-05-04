#pragma once

#include <iostream>

enum KERNEL_VERSION {
  ALPHA = 0,
  BETA = 1,
  DELTA = 2,
  EPSILON = 3,
  ZETA = 4,
  ETA = 5,
};

KERNEL_VERSION str2version(std::string version_str) {
  if (version_str == "alpha" || "ALPHA") {
    return KERNEL_VERSION::ALPHA;
  } else if (version_str == "beta" || version_str == "BETA") {
    return KERNEL_VERSION::BETA;
  } else if (version_str == "delta" || version_str == "DELTA") {
    return KERNEL_VERSION::DELTA;
  } else if (version_str == "epsilon" || version_str == "EPSILON") {
    return KERNEL_VERSION::EPSILON;
  } else if (version_str == "eta" || version_str == "ETA") {
    return KERNEL_VERSION::ETA;
  } else {
    std::cerr << "The version string is incorrect." << std::endl;
    std::exit(-1);
  }
}