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
    options->ParseOptions(argc, argv);
    cout << "Options/faces: " << options->haar_faces() << endl;
    cout << "Options/eyes: " << options->haar_eyes() << endl;
    cout << "Options/data: " << options->user_data() << endl;
    cout << endl;
  }

  // Create cascade classifier: default face Haar Cascade resides in
  // /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
  CascadeClassifier haar_cascade;
  haar_cascade.load(options->haar_faces());

  // Create extra cascade for eyes
  CascadeClassifier haar_extra_eyes;
  haar_extra_eyes.load(options->haar_eyes());

  // Open video capture device
  VideoCapture cap(0); // 0 - default video capture device
  if (!cap.isOpened()) {
    cerr << "Capture device ID 0 cannot be opened." << endl;
  }

  // Process video input per frame
  Mat frame;
  while (true) {
    // Get current frame, convert it to gray to detect a face
    cap >> frame;
    Mat original = frame.clone();
    Mat gray;
    cvtColor(original, gray, CV_BGR2GRAY); // convert the frame
    vector<Rect_<int>> faces;
    double face_side = sqrt(original.cols * original.rows);
    // Detect a face (5..55% of a frame)
    haar_cascade.detectMultiScale(gray, faces, 1.2, 4,
        CASCADE_SCALE_IMAGE,
        Size(face_side * 0.05, face_side * 0.05),
        Size(face_side * 0.55, face_side * 0.55));

    // Now faces variable holds rectangles for all detected faces
    // in the current frame. Annotate every face in the frame
    if (faces.size() == 1) {
      Rect face_rect = faces[0];
      vector<Rect_<int>> eyes;
      // Detect eyes (10..30% of a face)
      haar_extra_eyes.detectMultiScale(Mat(gray, face_rect), eyes, 1.2, 3,
          CASCADE_SCALE_IMAGE,
          Size(face_rect.width * 0.1, face_rect.height * 0.1),
          Size(face_rect.width * 0.3, face_rect.height * 0.3));
      if (eyes.size() == 2) { // early reject false face
        Rect eyes_line = eyes[0] | eyes[1];
        if (eyes_line.width > 2 * eyes_line.height) { // false eyes
          // Highlight the face with a rectangle and a text in the frame
          rectangle(original, face_rect, CV_RGB(0, 255, 0), 1);
          rectangle(original, eyes_line + face_rect.tl(), CV_RGB(0, 0, 255), 1);

          string box_text = format("Face: %dx%d px",
              face_rect.width, face_rect.height);

          int pos_x = std::max(face_rect.tl().x - 10, 0);
          int pos_y = std::max(face_rect.tl().y - 10, 0);

          putText(original, box_text, Point(pos_x, pos_y),
              FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);
          // Display the frame with face detected
          imshow("facelock", (original));
        }
      }
    }

    // Wait and catch keypress
    char key = (char) waitKey(80);
    if (key == 27) break;
  }

  destroyAllWindows();
  delete options;
  return 0;
}

// vim: se et ts=2 sw=2 number:
