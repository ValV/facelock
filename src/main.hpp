// Header file for main.cpp

#include <sys/stat.h>

inline bool exists(const std::string &path) {
  struct stat buffer;
  return (stat (path.c_str(), &buffer) == 0);
}

// vim: se et ts=2 sw=2 number:
