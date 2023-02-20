#ifndef TRTENGINE_H
#define TRTENGINE_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "logging.h"
#include <QString>
#include <QSize>
#include <QJsonObject>
#include <QJsonArray>
#include <QJsonValue>
#include <QStringList>
#include <QFile>
#include <QJsonParseError>
#include <QJsonDocument>
#include <QFileInfo>
#include <QTimer>
#include <QEventLoop>
#include <QThread>

using namespace nvinfer1;


QJsonObject readJson(QString json_file);
void dumpJson(QJsonObject obj, QString file_name);
cv::Mat draw(cv::Mat image, QJsonObject objects, int draw_size, bool draw_label);

class TensorRTEngine {
    int num_classes=0;
    int output_num_array=0;

    double conf_thres = 0.25;
    double nms_thres = 0.45;
    long long output_size = 1;
    QSize image_size = QSize(640, 640);
    size_t size{0};
    QStringList class_names = QStringList();

    IRuntime* runtime;
    IExecutionContext* context;
    ICudaEngine* engine;
//    ICudaEngine& infer_engine;

    int num_objects=0;
    int inputIndex;
    int outputIndex;
    int batch=1;

    bool first=true;
    bool cuda_occupyed=false;
    bool pixel_range1=false;
    bool obj_conf_enabled=true;

    QString input_name = "input_0";
    QString output_name = "output_0";
    cudaStream_t stream;
    void* buffers[2];
    float* prob;

public:
    TensorRTEngine(QString engine_file, QSize input_size);
    TensorRTEngine();

    QString load_engine(QString cfg_file);
    bool engine_loaded=false;
    void load_classes(QString class_file);
    void load_classes(QJsonArray classes);

    void set_conf_threshold(float thres);
    void set_nms_threshold(float thres);
    void set_input_name(QString name);
    void set_output_name(QString name);
    void release();

    QJsonObject inference(cv::Mat image);
    int get_number_objects();

};

class Detector : public QObject
{
    Q_OBJECT
    QThread *timerThread = new QThread;
public:
    cv::VideoCapture cap;
    TensorRTEngine detector;
    cv::Mat img;
    cv::Mat img_show;
    bool opened = true;
    bool updated = false;
    QTimer *t = new QTimer();

    int flag = -1;
    bool draw_label = true;
    bool loop=false;

    Detector(cv::VideoCapture cap, TensorRTEngine detector);
    ~Detector();
    void release();
    void start();
    bool isRunning();

private slots:

    void infer_and_show();
};

#endif
#define TRTENGINE_H
