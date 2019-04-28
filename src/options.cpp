#include <getopt.h>
#include <iostream>
#include <string>

#include "options.hpp"

const char *ProgramOptions::kHaarFaces = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
const char *ProgramOptions::kHaarEyes = "/usr/share/opencv4/haarcascades/haarcascade_eye.xml";
const char *ProgramOptions::kUserData = "data";

const struct option ProgramOptions::long_options_[] = {
  {"faces", required_argument, 0, 'f'},
  {"eyes", required_argument, 0, 'e'},
  {"train", no_argument, 0, 't'},
  {0, 0, 0, 0}
};

void ProgramOptions::init() {
  this->haar_faces_ = new std::string(this->kHaarFaces);
  this->haar_eyes_ = new std::string(this->kHaarEyes);
  this->user_data_ = new std::string(this->kUserData);
}

ProgramOptions::ProgramOptions() {
  this->init();
}

ProgramOptions::ProgramOptions(const std::string name) {
  this->init();
  this->user_data_->insert(0, name, 0, name.rfind('/') + 1);
}

ProgramOptions::~ProgramOptions() {
  if ((char *) this->haar_faces_ != this->kHaarFaces) {
    delete this->haar_faces_;
  }
  if ((char *) this->haar_eyes_ != this->kHaarEyes) {
    delete this->haar_eyes_;
  }
  if ((char *) this->user_data_ != this->kUserData) {
    delete this->user_data_;
  }
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

const std::string ProgramOptions::haar_faces(const char *path) {
  if (exists(path)) {
    if ((char *) this->haar_faces_ != this->kHaarFaces) {
      delete this->haar_faces_;
    }
    this->haar_faces_ = new std::string(path);
  }
  return *this->haar_faces_;
}

const std::string ProgramOptions::haar_eyes() {
  return *this->haar_eyes_;
}

const std::string ProgramOptions::haar_eyes(const char *path) {
  if (exists(path)) {
    if ((char *) this->haar_eyes_ != this->kHaarEyes) {
      delete this->haar_eyes_;
    }
    this->haar_eyes_ = new std::string(path);
  }
  return *this->haar_eyes_;
}

const std::string ProgramOptions::user_data() {
  return *this->user_data_;
}

const std::string ProgramOptions::user_data(const char *path) {
  if (exists(path)) {
    if ((char*) this->user_data_ != this->kUserData) {
      delete this->user_data_;
    }
    this->user_data_ = new std::string(path);
  }
  return *this->user_data_;
}

const int ProgramOptions::ParseOptions(int argc, char **argv) {
  int status = kParseStatusOk;

  int code = 0;
  int index = 0;
  while (true) {
    code = getopt_long(argc, argv, "f:e:t", this->long_options_, &index);
    if (code == -1) break;

    // Process option arguments
    switch(code) {
      case 0:
        // TODO: remove or signalize
        std::cout << "option " << this->long_options_[index].name;
        if (optarg) std::cout << " with arg " << optarg;
        std::cout << std::endl;
        break;
      case 'f':
        this->haar_faces(optarg);
        break;
      case 'e':
        this->haar_eyes(optarg);
        break;
      case 't':
        this->train(true);
        break;
      default:
        std::cout << "Note: getopt returned character code " << code;
        std::cout << std::endl;
        status |= kParseStatusExcessOption;
    }
  }

  // Process non-option arguments
  if (optind < argc) {
    if (optind < argc - 1) status |= kParseStatusExcessNonOption;
    while (optind < argc) {
      index = optind ++;
      this->user_data(argv[index]);
      if (exists(argv[index])) break; // TODO: remove, rewrite this loop
    }
  }

  return status;
}

void ProgramOptions::PrintSynopsis(const std::string name) {
  std::cout << "Usage: " << name.substr(name.rfind('/') + 1);
  std::cout << " [-f|--faces  <path/to/haar_faces>]";
  std::cout << " [-e|--eyes <path/to/haar_eyes>]";
  std::cout << " [-t|--training]";
  std::cout << " [path/to/data]" << std::endl;
}

// vim: se et ts=2 sw=2 number:
