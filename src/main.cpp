#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/videoio/videoio.hpp"

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char *argv[]) {
  // Check arguments
  if (argc < 2) {
    cout << "Usage: " << argv[0] << " <path/to/haar_cascade>" << endl;
    exit(1);
  }

  // Create cascade classifier (Haar)
  string haar_path = string(argv[1]);
  CascadeClassifier haar_cascade;
  haar_cascade.load(haar_path);

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
    cvtColor(original, gray, CV_BGR2GRAY); // convert the current frame
    vector<Rect_<int>> faces;
    haar_cascade.detectMultiScale(gray, faces); // detect faces from gray

    // Now faces variable holds rectangles for all detected faces
    // in the current frame. Annotate every face in the frame
    for (int i = 0; i < faces.size(); i ++) {
      // Highlight the face with a rectangle and a text in the frame
      Rect the_face_rect = faces[i];
      rectangle(original, the_face_rect, CV_RGB(0, 255, 0), 1);
      string box_text = format("Face found");

      int pos_x = std::max(the_face_rect.tl().x - 10, 0);
      int pos_y = std::max(the_face_rect.tl().y - 10, 0);

      putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN,
          1.0, CV_RGB(0, 255, 0), 2.0);
    }
    // Display the frame with faces detected
    imshow("face_recognizer", original);

    // Wait and catch keypress
    char key = (char) waitKey(50);
    if (key == 27) break;
  }

  destroyAllWindows();
  return 0;
}

// vim: se et ts=2 sw=2 number:
