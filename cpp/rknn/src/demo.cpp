#include "image_utils/rknn.h"
#include "argparse/argparser.h"
#include "print_utils.h"

#define DEFAULT_CONF_THRES 0.25
#define DEFAULT_NMS_THRES 0.45


argsutil::argparser get_args(int argc, char** argv) {
    argsutil::argparser parser("EdgeYOLO RKNN demo parser");
    parser.add_option<std::string>("-m", "--model", "model path", "model/edgeyolo_tiny_lrelu_coco.rknn");
    parser.add_option<bool>("-v", "--video", "use video", false);
    parser.add_option<bool>("-d", "--device", "use camera", false);
    parser.add_option<bool>("-p", "--picture", "use picture", false);
    parser.add_option<bool>("-nl", "--no-label", "do not draw label", false);
    parser.add_option<bool>("-l", "--loop", "loop", false);
    parser.add_option<std::string>("-s", "--source", "video source path", "0");
    parser.add_option<double>("-c", "--confidence-thres", "confidence threshold", DEFAULT_CONF_THRES);
    parser.add_option<double>("-n", "--nms-thres", "nms threshold", DEFAULT_NMS_THRES);

    parser.add_help_option();
    parser.parse(argc, argv);
    return parser;
}


int main(int argc, char** argv) {
    auto args = get_args(argc, argv);

    auto yoloNet = RKNN::YOLO(args.get_option_string("--model"), 
                              args.get_option_double("--confidence-thres"),
                              args.get_option_double("--nms-thres"));
    

    int ret = yoloNet.load_model();
    if (ret < 0) {
        printf("rknn model load failed with error code %d", ret);
        return ret;
    }
    detect::names = yoloNet.names;

    cv::VideoCapture cap;
    cv::Mat image, _img;
    bool draw_label = !args.get_option_bool("--no-label");
    bool loop = args.get_option_bool("--loop");

    if (args.get_option_bool("--video")) cap.open(args.get_option_string("--source"));
    else if (args.get_option_bool("--device")) cap.open(std::stoi(args.get_option_string("--source")));
    else if (args.get_option_bool("--picture")) image = cv::imread(args.get_option_string("--source"));

    struct timeval t0, t1;
    int delay=1;
    int count=0;
    int total_time=0;
    while (cap.isOpened()) {
        
        if (!cap.read(_img)) {
            if (loop) {
                cap.release();
                if (args.get_option_bool("--video")) cap.open(args.get_option_string("--source"));
                else if (args.get_option_bool("--device")) cap.open(std::stoi(args.get_option_string("--source")));
                else if (args.get_option_bool("--picture")) image = cv::imread(args.get_option_string("--source"));
                continue;
            }
            break;
        }

        if (_img.empty()) break;

        
        gettimeofday(&t0, NULL);
        std::vector<detect::Object> objs = yoloNet.infer(_img);
        gettimeofday(&t1, NULL);

        int time_interval = get_time_interval(t0, t1);
        total_time += time_interval;
        count++;

        INFO << "inference time(include preprocess and postprocess): " << time_interval << "ms,  num objects:" << objs.size() << ENDL;
        
        cv::imshow("edgeyolo RKNN demo", detect::draw_boxes(_img, objs, 20, draw_label));
        
        int key = cv::waitKey(delay);
        if (key==27) {
            cap.release();
            cv::destroyAllWindows();
            break;
        }
        else if (key==(int)(' ')) delay = 1 - delay;

    }
    INFO << "AVERAGE time delay: " << (float)total_time / count << " ms" << ENDL;

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