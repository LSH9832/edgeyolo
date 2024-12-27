#ifndef YOLO_DEMO_IMAGE_CPP
#define YOLO_DEMO_IMAGE_CPP

#include "../yolo/detect.hpp"
#include "../yolo/os.h"
#include "../yolo/datetime.h"
#include "../yolo/str.h"
#include "./argparser.h"
#include "yaml-cpp/yaml.h"


argparser::ArgumentParser parseArgs(int argc, char** argv)
{
    argparser::ArgumentParser parser("yolo demo parser", argc, argv);

    parser.add_option<std::string>("-c", "--config", "config file path", "");
    parser.add_option<std::string>("-s", "--source", "images source, yaml", "");
    parser.add_option<int>("-t", "--times", "infer times", 1);
    parser.add_help_option();

    return parser.parse();
}


int main(int argc, char** argv)
{
    auto args = parseArgs(argc, argv);

    pystring configPath = args.get_option_string("--config");
    pystring sourcePath = args.get_option_string("--source");
    int times = args.get_option_int("--times");

    if (!configPath.lower().endswith(".yaml") && !configPath.lower().endswith("yml"))
    {
        std::cerr << "[E] only support yaml config file, got " << configPath.lower() << std::endl;
        return -1;
    }
    if (!os::path::isfile(configPath))
    {
        std::cerr << "[E] config file not exist!" << std::endl;
        return -1;
    }

    std::cout << "----------------start init detector-----------------" << std::endl;


    Detector det;
    if(!det.init(configPath.str()))
    {
        std::cerr << "[E] failed to init detector!"  << std::endl;
        return -1;
    }

    std::cout << "\033[32m\033[1m[INFO] acceleration platform: " 
              << det.platform() << "\033[0m" << std::endl;
    

    std::cout << "--------------------load image path list--------------------" << std::endl;
    auto cfg = YAML::LoadFile(sourcePath.str());

    auto dirPath = os::path::dirname(sourcePath);
    auto filenames = cfg["frames"].as<std::vector<std::string>>();

    std::vector<std::vector<float>> result;

    os::makedirs("./result");
    
    std::cout << "--------------------start detect--------------------" << std::endl;

    
    cv::Mat frame;
    for(auto filename:filenames)
    {
        std::string fp = os::path::join({dirPath, filename});
        frame = cv::imread(fp);
        if (frame.empty()) continue;

        double t0 = pytime::time();
        // ----------- detect -----------

        // std::cout << "detect once" << std::endl;
        result.clear();
        det.detect(frame, result);

        
        // std::cout << "detect once end" << std::endl;
        // det.preProcess(frame, resizedImage, ratio);

        // double t1 = pytime::time();

        // det.infer(resizedImage, ratio, result);

        double t2 = pytime::time();

        // ------------------------------
        // std::cerr << "\rdelay: " << std::fixed << std::setprecision(3) << 1000 * (t2 - t0) << "ms, "
        //           << " preprocess:" << 1000 * (t1 - t0) << "ms,  infer:" << 1000 * (t2 - t1) << " ms.   ";

        std::cerr << "\rdelay: " << std::fixed << std::setprecision(3) << 1000 * (t2 - t0) << "ms  ";

        cv::Mat frame_show;
        if (result.size())
        {
            frame = det.draw(frame, result);
        }
        cv::imwrite("result/" + filename, frame);

    }
    
    std::cout << std::endl;
    return 0;
}

#endif