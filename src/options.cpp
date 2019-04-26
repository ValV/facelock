//#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <string>

#include "options.hpp"

void ProgramOptions::init() {
  this->haar_faces_ = new std::string(this->kHaarFaces);
  this->haar_eyes_ = new std::string(this->kHaarEyes);
  this->user_data_ = new std::string(this->kUserData);
}

const char *ProgramOptions::kHaarFaces = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
const char *ProgramOptions::kHaarEyes = "/usr/share/opencv4/haarcascades/haarcascade_eye.xml";
const char *ProgramOptions::kUserData = "data";

ProgramOptions::ProgramOptions() {
  this->init();
}
ProgramOptions::ProgramOptions(const std::string name) {
  this->init();
  this->user_data_->insert(0, name, 0, name.rfind('/') + 1);
}
ProgramOptions::~ProgramOptions() {
  if (this->haar_faces_ != NULL) delete this->haar_faces_;
  if (this->haar_eyes_ != NULL) delete this->haar_eyes_;
  if (this->user_data_ != NULL) delete this->user_data_;
}

bool ProgramOptions::train() {
  return this->train_;
  }
  bool ProgramOptions::train(const bool train) {
this->train_ = train;
  return this->train_; // TODO: switch to enum (trainig positive|negative)
  }

const std::string ProgramOptions::haar_faces() {
  return *this->haar_faces_;
}
const std::string ProgramOptions::haar_faces(std::string &path) {
  if (exists(path)) {
    if (this->haar_faces_ != NULL) delete this->haar_faces_;
    this->haar_faces_ = &path;
  }
  return *this->haar_faces_;
}

const std::string ProgramOptions::haar_eyes() {
  return *this->haar_eyes_;
}
const std::string ProgramOptions::haar_eyes(const std::string &path) {
  if (exists(path)) {
    if (this->haar_eyes_ != NULL) delete this->haar_eyes_;
    this->haar_eyes_ = &const_cast<std::string&>(path);
  }
  return *this->haar_eyes_;
}

const std::string ProgramOptions::user_data() {
  return *this->user_data_;
}
const std::string ProgramOptions::user_data(std::string &path) {
  if (exists(path)) {
    if (this->user_data_ != NULL) delete this->user_data_;
    this->user_data_ = &path;
  }
  return *this->user_data_;
}

void ProgramOptions::ParseOptions(int argc, char **argv) {
  const struct option long_options_[] = {
    {"faces", required_argument, 0, 'f'},
    {"eyes", required_argument, 0, 'e'},
    {"train", no_argument, 0, 't'},
    {0, 0, 0, 0}
  };

  int code = 0;
  int index = 0;
  while (true) {
    code = getopt_long(argc, argv, "f:e:t", long_options_, &index);
    if (code == -1) break;

    // Process option arguments
    switch(code) {
      case 0:
        std::cout << "option " << long_options_[index].name;
        if (optarg) std::cout << " with arg " << optarg;
        std::cout << std::endl;
        break;
      case 'f':
        this->haar_faces(*(new std::string(optarg)));
        break;
      case 'e':
        this->haar_eyes(*(new std::string(optarg)));
        break;
      case 't':
        this->train(true);
        break;
      default:
        std::cout << "Note: getopt returned character code " << code << std::endl;
    }
  }

  // Process non-option arguments
  if (optind < argc) {
    std::cout << "Non-option arguments: ";
    while (optind < argc) {
      index = optind ++;
      this->user_data(*(new std::string(argv[index])));
      if (exists(argv[index])) break; // TODO: remove, rewrite this loop
    }
  }
}

void ProgramOptions::PrintSynopsis(const std::string name) {
  std::cout << "Usage: " << name.substr(name.rfind('/') + 1);
  std::cout << " [-f|--faces  <path/to/haar_faces>]";
  std::cout << " [-e|--eyes <path/to/haar_eyes>]";
  std::cout << " [-t|--training]";
  std::cout << " [path/to/data]" << std::endl;
}

// vim: se et ts=2 sw=2 number:
