#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/videoio/videoio.hpp"

#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <string>

#include "options.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  // Configure program options
  ProgramOptions *options = new ProgramOptions(argv[0]);
  if (options == NULL) {
    cout << " Failed to initialize program options. Terminating..." << endl;
    exit(-1);
  } else {
    cout << "Options/faces: " << options->haar_faces() << endl;
    cout << "Options/eyes: " << options->haar_eyes() << endl;
    cout << "Options/data: " << options->user_data() << endl;
    cout << endl;
    options->ParseOptions(argc, argv);
    cout << "Options/faces: " << options->haar_faces() << endl;
    cout << "Options/eyes: " << options->haar_eyes() << endl;
    cout << "Options/data: " << options->user_data() << endl;
    cout << endl;
  }

  // Create cascade classifier: default face Haar Cascade resides in
  // /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
  string haar_path_face = options->haar_faces();
  if (argc > 1 && exists(argv[1])) {
    haar_path_face = string(argv[1]);
  }
  CascadeClassifier haar_cascade;
  haar_cascade.load(haar_path_face);

  // Create extra cascade for eyes
  string haar_path_eyes = options->haar_eyes();
  CascadeClassifier haar_extra_eyes;
  haar_extra_eyes.load(haar_path_eyes);

  // Open video capture device
  VideoCapture cap(0); // 0 - default video capture device
  if (!cap.isOpened()) {
    cerr << "Capture device ID 0 cannot be opened." << endl;
  }

  // Process video input per frame
  Mat frame;
  while (true) {
    // Get current frame, convert it to gray and detect a face
    cap >> frame;
    Mat original = frame.clone();
    Mat gray;
    cvtColor(original, gray, CV_BGR2GRAY); // convert the frame
    vector<Rect_<int>> faces;
    haar_cascade.detectMultiScale(gray, faces); // detect faces from gray

    vector<Rect_<int>> eyes;
    haar_extra_eyes.detectMultiScale(gray, eyes);

    // Now faces variable holds rectangles for all detected faces
    // in the current frame. Annotate every face in the frame
    // (if a face is between 5% and 55% of a frame)
    if (eyes.size() == 2) { // early reject false face
      Rect eyes_line = eyes[0] | eyes[1];
      if (eyes_line.width > 2 * eyes_line.height) {
        // Highlight the face with a rectangle and a text in the frame
        for (int i = 0; i < faces.size(); i ++) {
          Rect face_rect = faces[i];
          double face_ratio = face_rect.width * face_rect.height * 100.0 /
            original.cols / original.rows;
          // Filter out bad frames (face must be 5..55% of a frame)
          if (!((face_rect & eyes_line) == eyes_line)) break;
          if ((5 >= face_ratio) || (55 <=face_ratio)) break;
          // Good face rectangle (ready for training or evaluating)
          // Draw rectangles and labels (TODO: make it as an option)
          rectangle(original, face_rect, CV_RGB(0, 255, 0), 1);
          rectangle(original, eyes_line, CV_RGB(0, 0, 255), 1);

          string box_text = format("Face area: %dx%d px (%.2f%%)",
              face_rect.width, face_rect.height, face_ratio);

          int pos_x = std::max(face_rect.tl().x - 10, 0);
          int pos_y = std::max(face_rect.tl().y - 10, 0);

          putText(original, box_text, Point(pos_x, pos_y),
              FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
        }
      }
    }
    // Display the frame with faces detected
    imshow("face_recognizer", original);

    // Wait and catch keypress
    char key = (char) waitKey(50);
    if (key == 27) break;
  }

  destroyAllWindows();
  delete options;
  return 0;
}

// vim: se et ts=2 sw=2 number:
