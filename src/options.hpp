// Header file for main.cpp

#ifndef FACELOCK_OPTIONS_H
#define FACELOCK_OPTIONS_H

#include <sys/stat.h>

inline bool exists(const std::string &path) {
  struct stat buffer;
  return (stat (path.c_str(), &buffer) == 0);
};

class ProgramOptions {
  private:
    bool train_ = false;
    std::string *haar_faces_;
    std::string *haar_eyes_;
    std::string *user_data_;

    static const struct option long_options_[];

    void init();

  public:
    static const char *kHaarFaces;
    static const char *kHaarEyes;
    static const char *kUserData;

    typedef enum ParseStatus {
      kParseStatusOk = 0x0000,
      kParseStatusExcessNonOption = 0x0001,
      kParseStatusExcessOption = 0x0002,
      kParseStatisWrongOption = 0x0004
    } ParseStatus;

    ProgramOptions();
    ProgramOptions(const std::string name);
    ~ProgramOptions();

    bool train();
    bool train(const bool train);

    const std::string haar_faces();
    const std::string haar_faces(const char *path);

    const std::string haar_eyes();
    const std::string haar_eyes(const char *path);

    const std::string user_data();
    const std::string user_data(const char *path);

    const int ParseOptions(int argc, char **argv);
    void PrintSynopsis(const std::string name);
};

#endif

// vim: se et ts=2 sw=2 number:
