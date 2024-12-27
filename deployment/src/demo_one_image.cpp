#ifndef YOLO_DEMO_ONE_IMAGE_CPP
#define YOLO_DEMO_ONE_IMAGE_CPP

#include "../yolo/detect.hpp"
#include "../yolo/os.h"
#include "../yolo/datetime.h"
#include "../yolo/str.h"
#include "./argparser.h"
#include "yaml-cpp/yaml.h"


argparser::ArgumentParser parseArgs(int argc, char** argv)
{
    argparser::ArgumentParser parser("yolo infer one image demo parser", argc, argv);

    parser.add_option<std::string>("-c", "--config", "config file path", "");
    parser.add_option<std::string>("-s", "--source", "image source config", "");
    parser.add_help_option();

    return parser.parse();
}


void writefile(pystring msg, pystring filepath) {
    std::ofstream outfile(filepath.c_str(), std::ios::app);  // 创建ofstream对象，并打开文件
    if (!outfile.is_open()) { // 检查文件是否成功打开
        std::cerr << "failed to open " << filepath << std::endl;
        return;
    }
    outfile << msg; // 写入文件
    outfile.close(); // 关闭文件
}


int main(int argc, char** argv)
{
    auto args = parseArgs(argc, argv);

    pystring configPath = args.get_option_string("--config");
    pystring sourcePath = args.get_option_string("--source");



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
    
    
    
    std::cout << "--------------------start detect--------------------" << std::endl;

    std::string last_image_path="";
    int count = 0;
    std::vector<std::string> imgPaths;
    while (1)
    {
        if (os::path::isfile(sourcePath))
        {
            pytime::sleep(1);
            YAML::Node imcfg;
            imcfg = YAML::LoadFile(sourcePath.str());
            
            imgPaths = imcfg["imgs"].as<std::vector<std::string>>();
            
            break;
        }
        else
        {
            pytime::sleep(0.1);
        }
    }

    for (pystring imgPath: imgPaths)
    {
        pystring jsonPath = imgPath.split(".")[0] + ".json";
        while (!os::path::isfile(imgPath))
        {
            pytime::sleep(0.01);
        }
        cv::Mat image;
        while (1)
        {
            image = cv::imread(imgPath.str());
            if (!image.empty())
            {
                break;
            }
            else
            {
                std::cout << "read image again" << std::endl;
                pytime::sleep(0.05);
            }
        }
        
        
        std::vector<std::vector<float>> result;
        float ratio=1.0;
        cv::Mat resizedImage;
        det.preProcess(image, resizedImage, ratio);
        det.infer(resizedImage, ratio, result);

        std::ostringstream oss;
        oss << "[";
        for (int i=0;i<result.size();i++)
        {
            if(i) oss << ",";
            
            oss << "{'category_id':" << (int)result[i][4] << ",'bbox':[" 
                << result[i][0] << "," << result[i][1] << "," 
                << result[i][2]-result[i][0] << "," << result[i][3] - result[i][1] << "],'score':"
                << result[i][5] << ",'segmentation':[]}";
        }

        oss << "]";

        writefile(oss.str(), jsonPath);
        // std::string command = "echo \"" + oss.str() + "\" > " + jsonPath.str();
        // system(command.c_str());

        std::cout << "detect No. " << ++count /* << ": " << oss.str() */ << std::endl;
    }





    return 0;
}

#endif