#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/videoio/videoio.hpp"

#include <sys/stat.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

inline bool exists(const std::string &path) {
  struct stat buffer;
  return (stat (path.c_str(), &buffer) == 0);
};

const char *keys =
  "{ help h       |     | Print help message. }"
  "{ face f       |/usr/share/opencv4/haarcascades"
  "/haarcascade_frontalface_default.xml|"
  "             Path to pretrained face detect model (.xml). }"
  "{ eyes e       |/usr/share/opencv4/haarcascades"
  "/haarcascade_eye.xml|"
  "             Path to pretrained eye detect model (.xml). }"
  "{ width x      | 224 | Affine map window width (x axis) from 32px. }"
  "{ height y     | 224 | Affine map window height (y axis) from 32px. }"
  "{ showpts s    |false| Show landmark points + auxiliary points. }"
  ;


int main(int argc, char *argv[]) {
  // Configure program command line options
  CommandLineParser parser(argc, argv, keys);
  parser.about("Face detection OpenCV application");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  try {
    CV_Assert(parser.get<int>("width") > 31);
    CV_Assert(parser.get<int>("height") > 31);
    CV_Assert(exists(parser.get<string>("face")));
    CV_Assert(exists(parser.get<string>("eyes")));
  } catch (...) {
    parser.printMessage();
    cout << "Error: wrong command line or missing files!" << endl;
    cout << parser.get<string>("face") << endl;
    return 1;
  }

  bool showpts = parser.get<bool>("showpts");
  int affineWidth = parser.get<int>("width");
  int affineHeight = parser.get<int>("height");
  float eyeDesired[] = {0.35, 0.35}; // desired left eye x, y shift

  // Open video capture device
  VideoCapture cap(0); // 0 - default video capture device
  if (!cap.isOpened()) {
    cerr << "Capture device ID 0 cannot be opened." << endl;
  }

  // Create Haar face cascade classifier
  CascadeClassifier haar_face;
  haar_face.load(parser.get<string>("face"));

  // Create extra cascade for eyes
  CascadeClassifier haar_eyes;
  haar_eyes.load(parser.get<string>("eyes"));

  // Process video input per frame
  Mat frame;
  double frames_total = 0, frames_face = 0, frames_eyes = 0;
  double time_face = 0, time_eyes = 0;
  clock_t time_start = 0;
  while (true) {
    // Get current frame, convert it to gray to detect a face
    cap >> frame;
    frames_total ++;
    Mat original = frame.clone();
    Mat gray;
    cvtColor(original, gray, COLOR_BGR2GRAY);
    vector<Rect_<int>> faces;
    double face_side = sqrt(original.cols * original.rows);
    // Detect a face (5..55% of a frame)
    time_start = clock();
    haar_face.detectMultiScale(gray, faces, 1.2, 4,
        CASCADE_SCALE_IMAGE,
        Size(face_side * 0.05, face_side * 0.05),
        Size(face_side * 0.55, face_side * 0.55));
    time_face = (double) (clock() - time_start) / CLOCKS_PER_SEC;

    // Now faces variable holds rectangles for all detected faces
    // in the current frame. Annotate a face in the frame
    if (faces.size() == 1) {
      frames_face ++;
      Rect face_rect = faces[0];
      vector<Rect_<int>> eyes;
      // Detect eyes (10..30% of a face)
      time_start = clock();
      haar_eyes.detectMultiScale(Mat(gray, face_rect), eyes, 1.2, 3,
          CASCADE_SCALE_IMAGE,
          Size(face_rect.width * 0.1, face_rect.height * 0.1),
          Size(face_rect.width * 0.3, face_rect.height * 0.3));
      time_eyes = (double) (clock() - time_start) / CLOCKS_PER_SEC;
      if (eyes.size() == 2) { // early reject false eyes
        Rect eyes_rect = eyes[0] | eyes[1];
        if (eyes_rect.width > 2 * eyes_rect.height) {
          frames_eyes ++;
          // Calculate a center of each eye and a center between eyes
          Point2f eyeLeft;
          Point2f eyeRight;
          if (eyes[0].x < eyes[1].x) { // eyes[0] - left, eyes[1] - right
            eyeLeft = face_rect.tl() + (eyes[0].tl() + eyes[0].br()) / 2;
            eyeRight = face_rect.tl() + (eyes[1].tl() + eyes[1].br()) / 2;
          } else {
            eyeLeft = face_rect.tl() + (eyes[1].tl() + eyes[1].br()) / 2;
            eyeRight = face_rect.tl() + (eyes[0].tl() + eyes[0].br()) / 2;
          }
          Point2f eyesCenter = (eyeLeft + eyeRight) / 2;
          // Calculate shifts, scale, and rotation angle
          float dX = eyeRight.x - eyeLeft.x;
          float dY = eyeRight.y - eyeLeft.y;
          float scale = (1.0 - eyeDesired[0] - eyeDesired[0]) * affineWidth
            / (sqrt(pow(dX, 2) + pow(dY, 2)));
          float angle = atan2(dY, dX) * 180 / M_PI;
          // Get affine rotation matrix
          Mat M = getRotationMatrix2D(eyesCenter, angle, scale);
          // Adjust eyes' center target point
          float tX = affineWidth * 0.5;
          float tY = affineHeight * eyeDesired[1];
          M.at<double>(0, 2) += (tX - eyesCenter.x);
          M.at<double>(1, 2) += (tY - eyesCenter.y);
          // Create affine transformation (map)
          Mat affine(affineWidth, affineHeight, CV_32F);
          warpAffine(original, affine, M, Size(affineWidth, affineHeight));
          // Draw face landmarks
          if (showpts) {
            // Left eye rectangle
            putText(original, "L", eyeLeft + Point2f(-4, -6),
                FONT_HERSHEY_PLAIN, 1.0, CV_RGB(250, 128, 114), 2.0);
            // Right eye rectangle
            putText(original, "R", eyeRight + Point2f(-4, -6),
                FONT_HERSHEY_PLAIN, 1.0, CV_RGB(250, 128, 114), 2.0);
            rectangle(original, eyes_rect + face_rect.tl(),
                CV_RGB(60, 179, 113));
            // Auxiliary mideye point
            circle(original, eyesCenter, 3, CV_RGB(221, 160, 221), FILLED);
          }
          // Highlight a face with a rectangle and eye marks
          rectangle(original, face_rect, CV_RGB(250, 128, 114));
          circle(original, eyeLeft, 3, CV_RGB(240, 230, 140), FILLED);
          circle(original, eyeRight, 3, CV_RGB(240, 230, 140), FILLED);
          // Draw border around affine map
          rectangle(affine, Rect(0, 0, affine.cols, affine.rows),
              CV_RGB(112, 128, 144), 1);
          // Draw affine map on the frame
          affine.copyTo(original(Rect(original.cols - affineWidth,
                original.rows - affineHeight, affineWidth, affineHeight)));
        }
      }
      // Highlight the face with a rectangle
      rectangle(original, face_rect, CV_RGB(46, 139, 87));
    }

    // Display detector performance (accuracy + time)
    string box_text = format("Accuracy: %3.1f%% (%3.1f%%), "
        "time: %.3fs (%.3fs)", frames_eyes / frames_total * 100,
        frames_face / frames_total * 100, time_eyes + time_face,
        time_face);
    putText(original, box_text, Point(2, 14),
        FONT_HERSHEY_PLAIN, 1.0, CV_RGB(199, 21, 133), 2.0);

    // Display the frame with face detected (and affine map applied)
    imshow("facelock", (original));

    // Wait and catch keypress
    char key = (char) waitKey(250);
    if (key == 27) break;
  }

  destroyAllWindows();
  return 0;
}

// vim: se et ts=2 sw=2 number:
