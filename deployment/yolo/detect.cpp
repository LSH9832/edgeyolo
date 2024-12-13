#ifndef YOLO_DETECT_CPP
#define YOLO_DETECT_CPP

#include "./detect.hpp"
#include "yaml-cpp/yaml.h"
#include "./os.h"
// #include "./datetime.h"

#include <dlfcn.h>


static bool checkDefined(YAML::Node node, std::vector<std::string> names) {
    bool ret = true;
    for (auto name: names) {
        if (!node[name].IsDefined()) {
            std::cerr << "[E] node named '" << name << "' is not defined." << std::endl;
            ret = false;
        }
    }
    return ret;
}


static void blobFromImagei8(cv::Mat& img, uchar* blob, bool normalize=false){
    // uchar* blob = new uchar[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    uchar range = normalize?255:1;
    for (size_t c = 0; int(c) < channels; c++)
        for (size_t  h = 0; int(h) < img_h; h++)
            for (size_t w = 0; int(w) < img_w; w++)
                blob[c * img_w * img_h + h * img_w + w] = (uchar)img.at<cv::Vec3b>(h, w)[c] / range;
}


static void blobFromImagef(cv::Mat& img, float* blob, bool normalize=false){
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    float range = normalize?255:1;
    for (size_t c = 0; int(c) < channels; c++)
        for (size_t  h = 0; int(h) < img_h; h++)
            for (size_t w = 0; int(w) < img_w; w++)
                blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / range;
}


static char** StringsToCharArr(const std::vector<std::string>& vec) {
    // Allocate an array of char* pointers, one for each string in the vector
    char** charArray = static_cast<char**>(malloc((vec.size() + 1) * sizeof(char*))); // +1 for NULL terminator
    if (!charArray) {
        throw std::bad_alloc();
    }

    // Populate the array with strdup'd copies of the strings in the vector
    for (size_t i = 0; i < vec.size(); ++i) {
        charArray[i] = strdup(vec[i].c_str());
        if (!charArray[i]) {
            for (size_t j = 0; j < i; ++j) {
                free(charArray[j]);
            }
            free(charArray);
            throw std::bad_alloc();
        }
    }

    charArray[vec.size()] = nullptr;
    return charArray;
}

static std::vector<pystring> getLibPath()
{
    std::vector<pystring> ret = {".", "./lib", "../lib", "/usr/lib", "/lib", "/usr/local/lib"};
    const char *result = std::getenv("LD_LIBRARY_PATH");
    std::vector<pystring> paths = pystring(std::string(result)).split(":");
    ret.insert(ret.end(), paths.begin(), paths.end());
    return ret;
}

static cv::Mat static_resize(const cv::Mat& img, cv::Size input_size, float& r) {
    if (img.size().width == input_size.width && img.size().height == input_size.height)
    {
        r = 1.0;
        return img;
    }
    r = std::min(input_size.width / (img.cols*1.0), input_size.height / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;

    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_size.height, input_size.width, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}


static const int color_list[][3] = {
    {255, 56, 56},{255, 157, 151},{255, 112, 31},{255, 178, 29},{207, 210, 49},
    {72, 249, 10},{146, 204, 23},{61, 219, 134},{26, 147, 52},{0, 212, 187},
    {44, 153, 168},{0, 194, 255},{52, 69, 147},{100, 115, 255},{0, 24, 236},
    {132, 56, 255},{82, 0, 133},{203, 56, 255},{255, 149, 200},{255, 55, 198}
};

static cv::Scalar get_color(int index){
    index %= 20;
    return cv::Scalar(color_list[index][2], color_list[index][1], color_list[index][0]);
}

// --------------------------------------------------------------


struct Detector::Impl
{
    void* dllHandle=nullptr;
    typedef void* (*setupYOLO_f)(
        const char* modelFile, const char* inputName, char** outputNames, 
        int lengthOutputs, int imgW, int imgH, 
        const int* strides, int length_strides, int device
    );
    typedef void (*setYOLO_f)(void* yolo, const char* key, const char* value);
    typedef bool (*initYOLO_f)(void* yolo);
    typedef bool (*isInit_f)(void* yolo);
    typedef int (*getNumClasses_f)(void* yolo);
    typedef int (*getNumArrays_f)(void* yolo);
    typedef void (*inference_f)(void* yolo, void* data, void* preds, float scale);
    typedef bool (*isInputReachable_f)(void* yolo);
    typedef void* (*getInputData_f)(void* yolo);
    typedef void (*releaseYOLO_f)(void* yolo);
    typedef void (*platform_f)(char* p);

/* ------------------------------------------------ */
    

    setupYOLO_f setupYOLO;
    setYOLO_f set;
    initYOLO_f initYOLO;
    isInit_f isInit;
    getNumClasses_f getNumClasses;
    getNumArrays_f getNumArrays;
    inference_f inference;
    releaseYOLO_f releaseYOLO;
    isInputReachable_f isInputReachable;
    getInputData_f getInputData;
    platform_f platform;

/* ------------------------------------------------ */
    pystring platform_;
    void* yolo_=nullptr;
    int numArrays=0;
    int numClasses=0;
    std::vector<std::string> names={};
    float* preds=nullptr;
    bool objConfEnabled=true;
    bool normalize=false;
    cv::Size sz_;
    bool transpose=true;  // no transpose must no normalize, please note this
    bool inputFloat=true;
    bool isInit_=false;

    bool inputDataReachable=false;

    void* blob=nullptr;

/* ------------------------------------------------ */

    bool setupByModelSuffix(pystring modelPath)
    {
        auto ss = modelPath.split(".");
        pystring suffix = ss[ss.size()-1].lower();
        pystring libName = "libyoloInfer";
        if (suffix == "engine" || suffix == "trt")
        {
            libName += "TensorRT.so";
        }
        else if (suffix == "mnn")
        {
            libName += "MNN.so";
        }
        else if (suffix == "rknn")
        {
            libName += "RKNN.so";
        }
        else if (suffix == "om")
        {
            libName += "Ascend.so";
        }
        else if (suffix == "bin")
        {
            libName += "Horizon.so";
        }
        else 
        {
            std::cerr << "unsupported format '" << suffix << "'" << std::endl;
            return false;
        }

        pystring libPath = "";
        for (pystring path: getLibPath())
        {
            pystring libPath_ = os::path::join({path, libName});
            if (os::path::isfile(libPath_))
            {
                libPath = libPath_;
                break;
            }
            libPath_ = os::path::join({path, "yolo", libName});
            if (os::path::isfile(libPath_))
            {
                libPath = libPath_;
                break;
            }
        }

        if (libPath.empty())
        {
            std::cerr << "[E] can not find '" << libName << "'" << std::endl;
            return false;
        }

        std::cout << "loading " << libPath << std::endl;
        // pytime::sleep(1);
        dllHandle = dlopen(libPath.c_str(), RTLD_LAZY);
        // std::cout << 0 << std::endl;
        if (!dllHandle) {
            std::cerr << "[E] failed to open '" << libName << "' -> " << dlerror() << std::endl;
            return false;
        }



        std::cout << "load lib end." << std::endl;
        setupYOLO = (setupYOLO_f)dlsym(dllHandle, "setupYOLO");
        if (!setupYOLO) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        // std::cout << 2 << std::endl;
        
        set = (setYOLO_f)dlsym(dllHandle, "set");
        if (!set) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        // std::cout << 3 << std::endl;
        initYOLO = (initYOLO_f)dlsym(dllHandle, "initYOLO");
        if (!initYOLO) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        // std::cout << 4 << std::endl;
        isInit = (isInit_f)dlsym(dllHandle, "isInit");
        if (!isInit) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        // std::cout << 5 << std::endl;
        getNumClasses = (getNumClasses_f)dlsym(dllHandle, "getNumClasses");
        if (!getNumClasses) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        // std::cout << 6 << std::endl;
        getNumArrays = (getNumArrays_f)dlsym(dllHandle, "getNumArrays");
        if (!getNumArrays) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        // std::cout << 7 << std::endl;
        inference = (inference_f)dlsym(dllHandle, "inference");
        if (!inference) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        // std::cout << 8 << std::endl;
        releaseYOLO = (releaseYOLO_f)dlsym(dllHandle, "releaseYOLO");
        if (!releaseYOLO) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }

        isInputReachable = (isInputReachable_f)dlsym(dllHandle, "isInputReachable");
        if (!isInputReachable) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        
        getInputData = (getInputData_f)dlsym(dllHandle, "getInputData");
        if (!getInputData) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }

        // std::cout << 9 << std::endl;
        platform = (platform_f)dlsym(dllHandle, "platform");
        if (!platform) {
            fprintf(stderr, "Error finding function: %s\n", dlerror());
            dlclose(dllHandle);
            return false;
        }
        // std::cout << 10 << std::endl;
        return true;
    }
};


bool Detector::init(std::string configFile)
{
    if (impl_ == nullptr)
    {
        impl_ = new Impl();
    }
    YAML::Node cfg = YAML::LoadFile(configFile);
    checkDefined(cfg, {
        "model_path",
        "img_size",
        "names",
        "input_name",
        "output_names",
        "obj_conf_enabled",
        "normalize",
        "kwargs"
    });

    pystring modelPath = cfg["model_path"].as<std::string>();

    // std::cout << 1 << std::endl;
    impl_->names = cfg["names"].as<std::vector<std::string>>();
    // std::cout << 2 << std::endl;
    impl_->normalize = cfg["normalize"].as<bool>();
    // std::cout << 3 << std::endl;
    impl_->objConfEnabled = cfg["obj_conf_enabled"].as<bool>();
    // std::cout << 4 << std::endl;

    if(!impl_->setupByModelSuffix(modelPath))
    {
        std::cerr << "[E] failed to load '" << modelPath << "'." << std::endl;
        return false;
    }

    std::vector<int> strides={8, 16, 32};
    if (cfg["strides"].IsDefined())
    {
        strides = cfg["strides"].as<std::vector<int>>();
    }
    int device = 0;
    if (cfg["device"].IsDefined())
    {
        device = cfg["device"].as<int>();
    }
    impl_->sz_ = cv::Size(cfg["img_size"][1].as<int>(), cfg["img_size"][0].as<int>());

    // std::cout <<  cfg["input_name"].as<std::string>() << std::endl;
    modelPath = os::path::join({os::path::dirname(configFile), modelPath}).c_str();
    
    std::string inputNames = cfg["input_name"].as<std::string>();
    
    auto outputNames = cfg["output_names"].as<std::vector<std::string>>();
    char** outputNames_c = StringsToCharArr(outputNames);

    // std::cout << 1 << std::endl;
    
    impl_->yolo_ = impl_->setupYOLO(
        modelPath.c_str(), 
        inputNames.c_str(),
        outputNames_c,
        outputNames.size(),
        impl_->sz_.width,
        impl_->sz_.height,
        strides.data(), strides.size(), device
    );

    // std::cout << 2 << std::endl;

    // set
    // std::cout << "set yolo" << std::endl;
    for (const auto& key_value : cfg["kwargs"]) {
        std::string key = key_value.first.as<std::string>();
        YAML::Node value = key_value.second;
        if (key == "anchors")
        {
            std::string anchorStr = "";
            auto anchors = value.as<std::vector<std::vector<std::vector<std::string>>>>();
            for(int i=0;i<anchors.size();i++)
            {
                if (i) anchorStr += ";";
                for (int j=0;j<anchors[i].size();j++)
                {
                    if (j) anchorStr += ",";
                    anchorStr += anchors[i][j][0] + " " + anchors[i][j][1];
                }
            }
            impl_->set(impl_->yolo_, key.c_str(), anchorStr.c_str());
        }
        else if (key == "rerank")
        {
            /* code */
            std::string rerankStr = "";
            auto rerank = value.as<std::vector<int>>();
            for(int i=0;i<rerank.size();i++)
            {
                if (i) rerankStr += ",";
                rerankStr += std::to_string(rerank[i]);
            }
            impl_->set(impl_->yolo_, key.c_str(), rerankStr.c_str());
        }
        
        else
        {
            // std::cout << key << ", " << value.as<std::string>().c_str() << std::endl;
            impl_->set(impl_->yolo_, key.c_str(), value.as<std::string>().c_str());
        }
    }
    

    // std::cout << "init yolo" << std::endl;
    if(!impl_->initYOLO(impl_->yolo_))
    {
        std::cerr << "failed to init yolo." << std::endl;
        return false;
    }

    impl_->inputDataReachable = impl_->isInputReachable(impl_->yolo_);

    std::cout << "inputdata reachable: " << (impl_->inputDataReachable?"true":"false") << std::endl;

    // std::cout << "get num arrays and num classes" << std::endl;
    impl_->numArrays = impl_->getNumArrays(impl_->yolo_);
    impl_->numClasses = impl_->getNumClasses(impl_->yolo_);

    // std::cout << impl_->numArrays << ", " << impl_->numClasses << std::endl;

    // std::cout << 4 << std::endl;
    // std::cout << "get platform" << std::endl;

    impl_->platform_ = this->platform();
    impl_->platform_ = impl_->platform_.lower();
    if (impl_->platform_ == "rknn")
    {
        impl_->normalize = false;
        impl_->transpose = false;
    }
    else
    {
        impl_->transpose = true;
    }

    if (impl_->inputDataReachable)
    {
        impl_->blob = impl_->getInputData(impl_->yolo_);    // ascend
    }
    else if (impl_->transpose && impl_->inputFloat)
    {
        impl_->blob = new float[impl_->sz_.area() * 3];
    }
    else
    {
        impl_->blob = new uchar[impl_->sz_.area() * 3];
    }

    for (int i=impl_->names.size();i<impl_->numClasses;i++)
    {
        impl_->names.push_back((pystring("unknown_") + (i+1)).c_str());
    }

    // std::cout << 7 << std::endl;
    // memset(impl_->preds, 0, impl_->numArrays * (impl_->numClasses + impl_->objConfEnabled?5:4) * sizeof(float));
    impl_->preds = new float[impl_->numArrays * (impl_->numClasses + 5)];
    // std::cout << 8 << std::endl;
    impl_->isInit_ = true;

    return true;
}


std::string Detector::platform()
{
    if (impl_ ==  nullptr)
    {
        std::cerr << "[W] detector not init! Platform is unknown!" << std::endl;
        return "unknown";
    }
    if (!impl_->isInit)
    {
        std::cerr << "[W] detector not init! Platform is unknown!" << std::endl;
        return "unknown";
    }
    if (!impl_->isInit(impl_->yolo_))
    {
        std::cerr << "[W] detector not init! Platform is unknown!" << std::endl;
        return "unknown";
    }

    char* platform_ = new char[64];
    memset(platform_, ' ', 64);
    impl_->platform(platform_);
    std::string ret = pystring(platform_).split(" ")[0];
    delete platform_;
    return ret;
}


void Detector::preProcess(const cv::Mat& image, cv::Mat& resized, float& ratio)
{
    if (!impl_->isInit_)
    {
        std::cerr << "[E] detector not init!" << std::endl;
        return;
    }
    if (!impl_->isInit(impl_->yolo_))
    {
        std::cerr << "[E] detector not init!" << std::endl;
        return;
    }
    resized = static_resize(image, impl_->sz_, ratio);
}


void Detector::infer(cv::Mat& image, float ratio, std::vector<std::vector<float>>& prediction)
{
    if (!impl_->isInit_)
    {
        std::cerr << "[E] detector not init!" << std::endl;
        return;
    }
    if (!impl_->isInit(impl_->yolo_))
    {
        std::cerr << "[E] detector not init!" << std::endl;
        return;
    }

    if (impl_->transpose)
    {
        if (impl_->inputFloat)
        {
            blobFromImagef(image, (float*)impl_->blob, impl_->normalize);
            impl_->inference(impl_->yolo_, impl_->blob, impl_->preds, ratio);
        }
        else
        {
            blobFromImagei8(image, (uchar*)impl_->blob, false);
            impl_->inference(impl_->yolo_, impl_->blob, impl_->preds, ratio);
        }
    }
    else
    {
        impl_->inference(impl_->yolo_, image.data, impl_->preds, ratio);
    }

    std::vector<float> obj(6);
    for (int i=0;i<(int)impl_->preds[0];i++)
    {
        int idx = i * 6;
        for(int j=0;j<6;j++)
        {
            obj[j] = impl_->preds[idx+j+1];
        }
        prediction.push_back(obj);
    }

}


void Detector::detect(const cv::Mat& image, std::vector<std::vector<float>>& prediction, cv::Size* oriSz)
{
    if (!impl_->isInit_)
    {
        std::cerr << "[E] detector not init!" << std::endl;
        return;
    }
    if (!impl_->isInit(impl_->yolo_))
    {
        std::cerr << "[E] detector not init!" << std::endl;
        return;
    }

    float ratio = 1.0;

    cv::Mat resizedImage = static_resize(image, impl_->sz_, ratio);

    if (oriSz != nullptr)
    {
        ratio = std::min(impl_->sz_.width / (oriSz->width*1.0), impl_->sz_.height / (oriSz->height*1.0));
    }

    if (impl_->transpose)
    {
        if (impl_->inputFloat)
        {
            blobFromImagef(resizedImage, (float*)impl_->blob, impl_->normalize);
            impl_->inference(impl_->yolo_, impl_->blob, impl_->preds, ratio);
        }
        else
        {
            blobFromImagei8(resizedImage, (uchar*)impl_->blob, false);
            impl_->inference(impl_->yolo_, impl_->blob, impl_->preds, ratio);
        }
    }
    else
    {
        impl_->inference(impl_->yolo_, resizedImage.data, impl_->preds, ratio);
    }

    std::vector<float> obj(6);
    for (int i=0;i<(int)impl_->preds[0];i++)
    {
        int idx = i * 6;
        for(int j=0;j<6;j++)
        {
            obj[j] = impl_->preds[idx+j+1];
        }
        prediction.push_back(obj);
    }
}


void Detector::set(std::string key, std::string value)
{
    if (!impl_->isInit_)
    {
        std::cerr << "[E] detector not init!" << std::endl;
        return;
    }
    if (!impl_->isInit(impl_->yolo_))
    {
        std::cerr << "[E] detector not init!" << std::endl;
        return;
    }
    impl_->set(impl_->yolo_, key.c_str(), value.c_str());
}


std::vector<std::string> Detector::getNames()
{
    if (impl_ == nullptr)
    {
        return {};
    }
    return impl_->names;
}


cv::Mat Detector::draw(
    cv::Mat& src,
    std::vector<std::vector<float>>& prediction,
    bool draw_label, int thickness
)
{
    cv::Mat dist = src.clone();

    cv::Scalar color;
    cv::Scalar txt_color;
    cv::Scalar txt_bk_color;

    cv::Size label_size;

    int baseLine = 0;
    int x1, y1, x2, y2, out_point_y;
    int line_thickness = std::round((double)thickness / 10.0);
    
    for(int k=0; k<prediction.size(); k++){
        int label_ = (int)prediction.at(k)[4];
        color = get_color(label_);

        x1 = prediction.at(k)[0];
        y1 = prediction.at(k)[1];
        x2 = prediction.at(k)[2];
        y2 = prediction.at(k)[3];

        cv::rectangle(
            dist,
            cv::Rect2f(x1, y1, x2-x1, y2-y1),
            color,
            line_thickness
        );
        
        if (draw_label){
            txt_color = (cv::mean(color)[0] > 127)?cv::Scalar(0, 0, 0):cv::Scalar(255, 255, 255);
            std::string label = impl_->names.at(label_) + " " + std::to_string(prediction.at(k)[5]).substr(0, 4);
            label_size = cv::getTextSize(label.c_str(), cv::LINE_AA, double(thickness) / 30.0, (line_thickness>1)?line_thickness-1:1, &baseLine);
            txt_bk_color = color; // * 0.7;
            y1 = (y1 > dist.rows)?dist.rows:y1 + 1;
            out_point_y = y1 - label_size.height - baseLine;
            if (out_point_y >= 0) y1 = out_point_y;
            cv::rectangle(dist, cv::Rect(cv::Point(x1 - (line_thickness - 1), y1), cv::Size(label_size.width, label_size.height + baseLine)),
                        txt_bk_color, -1);
            cv::putText(dist, label, cv::Point(x1, y1 + label_size.height),
                        cv::LINE_AA, double(thickness) / 30.0, txt_color, (line_thickness>1)?line_thickness-1:1);
        }

    }
    return dist;
}


Detector::~Detector()
{
    impl_->releaseYOLO(impl_->yolo_);
    if (dlclose(impl_->dllHandle) != 0) {
        std::cerr << "[E] failed to close dynamic link lib -> " << dlerror() << std::endl;
    }

    if (!impl_->inputDataReachable)
    {
        if (impl_->transpose && impl_->inputFloat)
        {
            delete static_cast<float*>(impl_->blob);
        }
        else
        {
            delete static_cast<uchar*>(impl_->blob);
        }
    }
    
    delete impl_->preds;
    impl_->blob = nullptr;
    impl_->preds = nullptr;
}



#endif