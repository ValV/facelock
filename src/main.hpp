// Header file for main.cpp

#include <sys/stat.h>

inline bool exists(const std::string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

// vim: se et ts=2 sw=2 number:
