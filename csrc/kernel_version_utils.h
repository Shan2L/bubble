#pragma once

#include <iostream>

int str2version(const std::string& version_str) {
  if (version_str == "alpha" || version_str == "ALPHA") {
    return 0;
  } else if (version_str == "beta" || version_str == "BETA") {
    return 1;
  } else if (version_str == "delta" || version_str == "DELTA") {
    return 2;
  } else if (version_str == "epsilon" || version_str == "EPSILON") {
    return 3;
  } else if (version_str == "eta" || version_str == "ETA") {
    return 4;
  } else {
    std::cerr << "The version string is incorrect." << std::endl;
    std::exit(-1);
  }
}