#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/face/facemark.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/dnn/dnn.hpp"

#include <unistd.h>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace cv::face;
using namespace cv::dnn;
using namespace std;

const char *keys =
  "{ help h    | | Print help message. }"
  "{ cascade c | | Path to pretrained face detect model. }"
  ;

int main(int argc, char *argv[]) {
  // Configure program options
  CommandLineParser parser(argc, argv, keys);
  parser.about("Face detection OpenCV application");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  float confidenceThreshold = 0.7; // initial confidence
  const string tensorflowConfigFile = "opencv_face_detector.pbtxt";
  const string tensorflowWeightFile = "opencv_face_detector_uint8.pb";
  Net mobileNet = readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);

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
    Mat inputBlob = blobFromImage(original, 1.0,
        Size(299, 299), Scalar(103.93, 116.77, 123.68),
        true, false);
    mobileNet.setInput(inputBlob, "data");
    Mat detection = mobileNet.forward("detection_out");
    Mat detectionMat(detection.size[2], detection.size[3],
        CV_32F, detection.ptr<float>());
    vector<Rect_<int>> faces;
    for (int i = 0; i < detectionMat.rows; i ++) {
      float confidence = detectionMat.at<float>(i, 2);
      if (confidence > confidenceThreshold) {
        int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * original.cols);
        int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * original.rows);
        int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * original.cols);
        int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * original.rows);
        faces.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
      }
    }

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
    char key = (char) waitKey(250);
    if (key == 27) break;
  }

  destroyAllWindows();
  return 0;
}

// vim: se et ts=2 sw=2 number:
