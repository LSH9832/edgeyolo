#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "fstream"
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <ctime>

#include "print_utils.h"
#include "image_utils/detect_process.h"
#include "image_utils/mnn.h"

#include "argparse/argparser.h"


argsutil::argparser get_args(int argc, char** argv) {
    argsutil::argparser parser("EdgeYOLO MNN demo parser");
    parser.add_option<std::string>("-c", "--cfg", "config path", "config/mnn_detection_coco.yaml");
    parser.add_option<bool>("-v", "--video", "use video", false);
    parser.add_option<bool>("-d", "--device", "use camera", false);
    parser.add_option<bool>("-p", "--picture", "use picture", false);
    parser.add_option<bool>("-nl", "--no-label", "do not show label", false);
    parser.add_option<bool>("-l", "--loop", "play in loop", false);

    parser.add_option<bool>("-cpu", "--cpu", "cpu mode, default auto", false);
    parser.add_option<bool>("-gpu", "--gpu", "gpu mode, default auto", false);

    parser.add_option<std::string>("-s", "--source", "video source path", "0");
    parser.add_help_option();
    parser.parse(argc, argv);
    return parser;
}


int main(int argc, char** argv) {
    auto args = get_args(argc, argv);
    
    bool show_label = !args.get_option_bool("--no-label");

    auto yoloNet = mnn_det::YOLO(args.get_option_string("--cfg"));
    yoloNet.set_device(args.get_option_bool("--cpu"), args.get_option_bool("--gpu"));
    if (!yoloNet.load_model()) {
        ERROR << "FAILED TO LOAD MNN MODEL " << yoloNet.mnn_model << ENDL;
        return -1;
    }

    detect::names = yoloNet.names;

    cv::VideoCapture cap;
    cv::Mat image, _img;

    if (args.get_option_bool("--video")) cap.open(args.get_option_string("--source"));
    else if (args.get_option_bool("--device")) cap.open(std::stoi(args.get_option_string("--source")));
    else if (args.get_option_bool("--picture")) image = cv::imread(args.get_option_string("--source"));

    struct timeval t0, t1;
    int delay=1;

    bool flag = true;
    bool loop = args.get_option_bool("--loop");

    while (cap.isOpened()) {
        if (!cap.read(_img)) flag=false;
        if (_img.empty()) flag=false;

        if (!flag) {
            if (loop) {
                cap.release();
                if (args.get_option_bool("--video")) cap.open(args.get_option_string("--source"));
                else if (args.get_option_bool("--device")) cap.open(std::stoi(args.get_option_string("--source")));
                flag=true;
                continue;
            }
            else break;
        }

        gettimeofday(&t0, NULL);
        std::vector<detect::Object> objs = yoloNet.infer(_img);
        gettimeofday(&t1, NULL);

        INFO << "inference time(include preprocess and postprocess): " << get_time_interval(t0, t1) << "ms,  num objects:" << objs.size() << ENDL;
        
        cv::imshow("edgeyolo MNN demo", detect::draw_boxes(_img, objs, 20, show_label));
        
        int key = cv::waitKey(delay);
        if (key==27) {
            cap.release();
            cv::destroyAllWindows();
            break;
        }
        else if (key==(int)(' ')) delay = 1 - delay;

    }

    if (!image.empty()) {
        gettimeofday(&t0, NULL);
        std::vector<detect::Object> objs = yoloNet.infer(image);
        gettimeofday(&t1, NULL);
        INFO << "inference time(include preprocess and postprocess): " << get_time_interval(t0, t1) << "ms,  num objects:" << objs.size() << ENDL;
        WARN << "press 's' to save this image" << ENDL;
        image = detect::draw_boxes(image, objs);
        cv::imshow("edgeyolo MNN demo", image);
        int key = cv::waitKey(0);
        if (key==(int)('s') || key==(int)('S')) {
            cv::imwrite("demo.jpg", image);
        };
    }

    return 0;
}
