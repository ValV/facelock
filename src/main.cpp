#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face/facemark.hpp"
#include "opencv2/videoio/videoio.hpp"

#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::face;
using namespace std;

const char *keys =
  "{ help h    | | Print help message. }"
  "{ cascade c |/usr/share/opencv4/lbpcascades/lbpcascade_frontalface.xml| Path to pretrained face detect model. }"
  ;

int main(int argc, char *argv[]) {
  // Configure program options
  CommandLineParser parser(argc, argv, keys);
  parser.about("Face detection OpenCV application");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  String cascade =  parser.get<String>("cascade");
  cout << "Path to cascade: " << parser.get<String>("cascade") << endl;

  // Create cascade classifier
  CascadeClassifier face_cascade;
  face_cascade.load(cascade);

  // Create face landmark predictor
  Ptr<Facemark> facemark = createFacemarkKazemi();
  facemark->loadModel("face_landmark_model.dat");

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
    equalizeHist(gray, gray);
    vector<Rect_<int>> faces;
    double face_side = sqrt(original.cols * original.rows);
    // Detect a face (5..55% of a frame)
    face_cascade.detectMultiScale(gray, faces, 1.2, 4,
        CASCADE_SCALE_IMAGE,
        Size(face_side * 0.07, face_side * 0.07),
        Size(face_side * 0.55, face_side * 0.55));

    // Now faces variable holds rectangles for all detected faces
    // in the current frame. Annotate every face in the frame
    if (faces.size() == 1) {
      Rect face_rect = faces[0];
      vector<vector<Point2f>> shapes;
      if (facemark->fit(original, faces, shapes)) {
        // Draw face landmarks
        for (int i = 36; i < 48; i ++) {
          circle(original, shapes[0][i], 2, Scalar(255, 0, 0), FILLED);
        }
      }
      // Highlight the face with a rectangle and a text in the frame
      rectangle(original, face_rect, CV_RGB(0, 255, 0), 1);
      // Display the frame with face detected
      imshow("facelock", (original));
    }

    // Wait and catch keypress (ESC)
    char key = (char) waitKey(100);
    if (key == 27) break;
  }

  destroyAllWindows();
  return 0;
}

// vim: se et ts=2 sw=2 number:
