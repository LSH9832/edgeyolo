#ifndef YOLO_HPP
#define YOLO_HPP


#include <iostream>
#include <string>
#include <vector>


class YOLO
{
public:

    YOLO() {}

    YOLO(std::string modelFile, std::string inputName, std::vector<std::string> outputNames,
         int imgW, int imgH, std::vector<int> strides, int device=0);

    bool init();

    void inference(void* data, void* preds, float scales=1.0);

    bool isInit();

    void set(std::string key, std::string value);

    bool inputDataReachable();

    void* getInputData();

    int getNumClasses();

    int getNumArrays();

    ~YOLO();

    struct Impl;
private:

    std::string modelFile_, inputName_;
    std::vector<std::string> outputNames_;
    int imgW_, imgH_;
    std::vector<int> strides_ = {8, 16, 32};
    
    Impl* impl_ = nullptr;
    
};

extern "C"
{
    YOLO* setupYOLO(
        const char* modelFile, const char* inputName, 
        char** outputNames, int lengthOutputs,
        int imgW, int imgH,
        const int* strides, int length_strides, int device
    )
    {
        std::vector<std::string> _outputNames;
        for(int i=0;i<lengthOutputs;i++)
        {
            // std::cout << outputNames[i] << std::endl;
            _outputNames.push_back(outputNames[i]);
        }

        std::vector<int> _strides;
        for (int i=0;i<length_strides;i++)
        {
            // std::cout << strides[i] << std::endl;
            _strides.push_back(strides[i]);
        }

        return new YOLO(modelFile, inputName, _outputNames, imgW, imgH, _strides, device);
    }

    void set(YOLO* yolo, const char* key, const char* value)
    {
        yolo->set(key, value);
    }


    bool initYOLO(YOLO* yolo)
    {
        return yolo->init();
    }

    bool isInit(YOLO* yolo)
    {
        return yolo->isInit();
    }

    int getNumClasses(YOLO* yolo)
    {
        return yolo->getNumClasses();
    }

    int getNumArrays(YOLO* yolo)
    {
        return yolo->getNumArrays();
    }

    void inference(YOLO* yolo, void* data, void* preds, float scale=1.0)
    {
        yolo->inference(data, preds, scale);
    }

    bool isInputReachable(YOLO* yolo)
    {
        return yolo->inputDataReachable();
    }

    void* getInputData(YOLO* yolo)
    {
        return yolo->getInputData();
    }

    void releaseYOLO(YOLO* yolo)
    {
        delete yolo;
        yolo=nullptr;
    }

    void platform(char* p);
}



#endif