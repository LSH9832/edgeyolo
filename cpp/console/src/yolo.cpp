#include <opencv2/opencv.hpp>
#include "image_utils/trt.h"
#include "argparse/argparser.h"
#include "print_utils.h"

using namespace std;

bool isnum(string s) {
    stringstream sin(s);
    double t;
    char p;
    if (!(sin >> t))
        return false;
    if (sin >> p)
        return false;
    else
        return true;
}

argsutil::argparser get_args(int argc, char* argv[]) {
    auto parser = argsutil::argparser("EdgeYOLO-TensorRT argument parser. Visit https://github.com/LSH9832/edgeyolo for more information.");

    /* example -> ./yolo.exe ./yolo.engine ./nuscenes_mini.mp4 --pause --loop --no-label */
    parser.add_argument<string>("engine", "engine file path")
          .add_argument<string>("source", "video source")
          .add_option<double>("-c", "--conf", "confidence threshold, default is 0.25", 0.25)
          .add_option<double>("-n", "--nms", "NMS threshold, default is 0.45", 0.45)
          .add_option<bool>("-nl", "--no-label", "do not draw labels. default is false", false)
          .add_option<bool>("-l", "--loop", "loop playback. default is false", false)
          .add_option<bool>("-p", "--pause", "pause playing at first. default is false", false)
          .add_help_option()
          .parse(argc, argv);
    
    return parser;
}

int main(int argc, char* argv[]) {
    
    auto args = get_args(argc, argv);

    std::string name = args.get_argument_string("engine");
    std::string source = args.get_argument_string("source");
    bool draw_label = !args.get_option_bool("--no-label");

    cv::VideoCapture cap;
    if (isnum(source))
        cap.open(atoi(source.c_str()));
    else
        cap.open(source);

    cudaSetDevice(0);

    /* -------- The following is the usage demo of yoloNet -------- */

    yoloNet *yolonet = new yoloNet();

    if (!yolonet->load_engine(name)) {
        cout << "failed to load engine" << endl;
        return -1;
    }

    yolonet->set_conf_threshold(args.get_option_double("--conf"));  /* optional function, confidence threshold, default is 0.25 */
    yolonet->set_nms_threshold(args.get_option_double("--nms"));    /* optional function, NMS threshold, default is 0.5 */

    cv::Mat frame;
    int key = -1;
    int delay = args.get_option_bool("--pause") ? 0 : 1;
    timeval t0, t1;
    while (cap.isOpened()) {
        cap >> frame;
        if (frame.empty()) {
            if (args.get_option_bool("--loop")) {
                if (isnum(source))
                    cap.open(atoi(source.c_str()));
                else
                    cap.open(source);
                continue;
            }
            else {
                cout << "frame is empty, exit." << endl;
                cv::destroyAllWindows();
                break;
            }
        }

        /**********************************************
        inference results: preds
        preds.size(): number of inference results
        let pred = preds.at(k), 0<=k<=preds.size() - 1

        pred.rect <cv::Rect>  : bounding box [x, y, w, h],  where x, y describe the location of left-top corner of the bounding box.
        pred.label <int>      : index of object's type
        pred.prob <double>    : confidence between 0-1
        ***********************************************/

        gettimeofday(&t0, NULL);

        std::vector<detect::Object> preds = yolonet->infer(frame);

        gettimeofday(&t1, NULL);

        printf("%sinfer: %d ms, number of objects: %d\n%s", GREEN, get_time_interval(t0, t1), preds.size(), END);

        /*************************************************/

        frame = detect::draw_boxes(frame, preds, yolonet->cfg.NAMES, 20, draw_label);

        cv::imshow("EdgeYOLO result", frame);
        key = cv::waitKey(delay);
        switch (key) {
        case 27:   // esc
            cap.release();
            break;
        case 32:   // space
            delay = 1 - delay;
            break;
        default:
            break;
        }
    }
    cv::destroyAllWindows();
    yolonet->release();
    return 0;
}
