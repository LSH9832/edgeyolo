#include <opencv2/opencv.hpp>
#include "trtengine.h"
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>


#define CHECK(status) \
    do\
    {\
        int ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.25

using namespace nvinfer1;
using namespace std;

// stuff we know about the network and the input/output blobs
//static const int NUM_CLASSES = 80;
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";
static Logger gLogger;

const int color_list[][3] = {
    {255, 56, 56},
    {255, 157, 151},
    {255, 112, 31},
    {255, 178, 29},
    {207, 210, 49},
    {72, 249, 10},
    {146, 204, 23},
    {61, 219, 134},
    {26, 147, 52},
    {0, 212, 187},
    {44, 153, 168},
    {0, 194, 255},
    {52, 69, 147},
    {100, 115, 255},
    {0, 24, 236},
    {132, 56, 255},
    {82, 0, 133},
    {203, 56, 255},
    {255, 149, 200},
    {255, 55, 198}
};


cv::Scalar get_color(int index){
    index = index % 20;
//    index -= (index==0)?0:1;
    return cv::Scalar(color_list[index][2], color_list[index][1], color_list[index][0]);
}

cv::Mat static_resize(cv::Mat& img, QSize input_size) {
    float r = std::min(input_size.width() / (img.cols*1.0), input_size.height() / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;

    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(input_size.height(), input_size.width(), CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(faceobjects[i], faceobjects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(faceobjects, left, j);
    if (i < right) qsort_descent_inplace(faceobjects, i, right);

}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = faceobjects[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_yolox_proposals(int num_array, float* feat_blob, float prob_threshold, std::vector<Object>& objects, int num_class)
{
    for (int anchor_idx = 0; anchor_idx < num_array; anchor_idx++)
    {
        const int basic_pos = anchor_idx * (num_class + 5);

        // yolox/models/yolo_head.py decode logic
        float x_center = feat_blob[basic_pos+0];
        float y_center = feat_blob[basic_pos+1];
        float w = feat_blob[basic_pos+2];
        float h = feat_blob[basic_pos+3];
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.prob = 0.0;

        float box_objectness = feat_blob[basic_pos+4];    // obj conf
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                if (box_prob > obj.prob) {
                    obj.label = class_idx;
                    obj.prob = box_prob;
                }
            }
        } // class loop
        if(obj.prob > prob_threshold) objects.push_back(obj);
    } // point anchor loop
}

static void generate_yolov8_proposals(int num_array, float* feat_blob, float prob_threshold, std::vector<Object>& objects, int num_class)
{
    int array_length = num_class + 4;
    for (int anchor_idx = 0; anchor_idx < num_array; anchor_idx++)
    {
        const int basic_pos = anchor_idx;

        float x_center = feat_blob[basic_pos + 0 * num_array];
        float y_center = feat_blob[basic_pos + 1 * num_array];
        float w = feat_blob[basic_pos + 2 * num_array];
        float h = feat_blob[basic_pos + 3 * num_array];
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.prob = 0.0;

        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_prob = feat_blob[basic_pos + (4 + class_idx) * num_array];
            if (box_prob > prob_threshold)
            {
                if (box_prob > obj.prob) {
                    obj.label = class_idx;
                    obj.prob = box_prob;
                }
            }
        } // class loop
        if(obj.prob > prob_threshold) objects.push_back(obj);
    } // point anchor loop
}

float* blobFromImage(cv::Mat& img, bool pixel_range1=false){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    float range = pixel_range1?255.0:1.0;
    for (size_t c = 0; int(c) < channels; c++)
        for (size_t  h = 0; int(h) < img_h; h++)
            for (size_t w = 0; int(w) < img_w; w++)
                blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / range;
    return blob;
}

static void decode_outputs(float* prob,
                           std::vector<Object>& objects,
                           float scale,
                           const int img_w,
                           const int img_h,
                           float conf_thres,
                           float nms_thres,
                           QSize img_size,
                           int num_class,
                           int num_array,
                           bool obj_conf_enabled=true) {
    std::vector<Object> proposals;

    if (obj_conf_enabled){
        generate_yolox_proposals(num_array, prob,  conf_thres, proposals, num_class);
    }
    else {
//        std::cout<<"yolov8"<<std::endl;
        generate_yolov8_proposals(num_array, prob,  conf_thres, proposals, num_class);
    }

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_thres);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

TensorRTEngine::TensorRTEngine() {
    cudaSetDevice(DEVICE);
    this->engine_loaded=false;
}

TensorRTEngine::TensorRTEngine(QString engine_file, QSize input_size) {
    cudaSetDevice(DEVICE);

    char *trtModelStream{nullptr};
    ifstream file(engine_file.toStdString(), ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        this->size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    this->image_size.setHeight(input_size.height());
    this->image_size.setWidth(input_size.width());

    this->runtime = createInferRuntime(gLogger);
    assert(this->runtime != nullptr);
    this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    this->context = engine->createExecutionContext();
    assert(this->context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);

    this->output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        this->output_size *= out_dims.d[j];
    }
    this->prob = new float[output_size];                //network output will be saved here
    CHECK(cudaStreamCreate(&stream));
    this->engine_loaded=true;
}

QString TensorRTEngine::load_engine(QString cfg_file) {
    // if engine is loaded, first release it and then load a new one.
    this->first = true;

    if (this->engine_loaded)
        this->release();
    this->engine_loaded = false;

    // read config from json file
    if(!cfg_file.endsWith(".json"))
        return QString("");
    QJsonObject cfg = readJson(cfg_file);
    if (!((cfg.contains("classes")||cfg.contains("names"))&&cfg.contains("img_size")&&cfg.contains("input_name")&&cfg.contains("output_name"))) {
        std::cout<<"json file does not contain all needed messages."<<std::endl;
        return QString("");
    }

    QJsonArray classes = cfg.value((cfg.contains("classes")?"classes":"names")).toArray();
    QJsonArray ipt_size = cfg.value("img_size").toArray();

    this->pixel_range1 = (cfg.contains("pixel_range") && (cfg.value("pixel_range").toInt() == 1));
    this->obj_conf_enabled = (cfg.contains("obj_conf_enabled") && cfg.value("obj_conf_enabled").toBool()) || !cfg.contains("obj_conf_enabled");


    std::cout<<"obj_conf_enabled: "<<this->obj_conf_enabled<< std::endl;

    QString engine_file = cfg_file.left(cfg_file.length()-5);
    engine_file.append(".engine");


    char *trtModelStream{nullptr};
    ifstream file(engine_file.toStdString(), ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        this->size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    this->image_size.setHeight(ipt_size.at(0).toInt());
    this->image_size.setWidth(ipt_size.at(1).toInt());

    this->runtime = createInferRuntime(gLogger);
    assert(this->runtime != nullptr);
    this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    this->context = engine->createExecutionContext();
    assert(this->context != nullptr);
    delete[] trtModelStream;

    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex(this->input_name.toStdString().c_str());
    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    outputIndex = engine->getBindingIndex(this->output_name.toStdString().c_str());
    assert(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);

    auto out_dims = engine->getBindingDimensions(inputIndex);

    this->output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {

        if (j==0) {
            int batch = out_dims.d[j];
            if (batch == -1){
                batch = engine->getMaxBatchSize();
            }
            if (this->batch > batch)
                this->batch = batch;
            this->output_size *= this->batch;
            out_dims.d[j] = this->batch;
            cout<<"batch size this time: "<<this->batch<<endl;

        }
        else this->output_size *= out_dims.d[j];
    }

    context->setBindingDimensions(inputIndex, out_dims);


    this->prob = new float[output_size];                //network output will be saved here

    this->num_classes = classes.size();
    this->class_names.clear();
    for(int idx=0; idx<this->num_classes; idx++) {
        this->class_names.append(classes.at(idx).toString());
    }

    this->output_num_array = output_size / ((this->obj_conf_enabled?5:4) + this->num_classes);

    this->set_input_name(cfg.value("input_name").toString());
    this->set_output_name(cfg.value("output_name").toString());

    CHECK(cudaStreamCreate(&stream));

//    this->engine_loaded=true;
    /* not support batch > 1 now */
    this->engine_loaded = this->batch == 1;
    if (!this->engine_loaded)
        return QString("[ERROR] Currently this program doesn't support model with batch > 1.");

    int max_length = 0;
    QString msg = QString();
    for(int idx=0; idx<classes.size(); idx++){
        max_length = (max_length>classes.at(idx).toString().length())?max_length:classes.at(idx).toString().length();
        msg.append(QString("%1\n").arg(classes.at(idx).toString()));
    }
    QString line = QString();
    while(max_length-->0)
        line.append("-");
    return QString("Number of classes: %1\n%2\n%3%4").arg(this->num_classes).arg(line).arg(msg).arg(line);
}

void TensorRTEngine::load_classes(QJsonArray classes){
    this->num_classes = classes.size();
    this->class_names.empty();
    for(int idx=0; idx<this->num_classes; idx++) {
        this->class_names.append(classes.at(idx).toString());
    }
    std::cout<<"number of classes: "<<this->num_classes<<std::endl;
    this->output_num_array = output_size / (5 + this->num_classes);

}

void TensorRTEngine::load_classes(QString class_file){
    ifstream in(class_file.toStdString());
    string line;
    this->class_names.empty();
    while (!in.eof()) {
        getline(in, line, '\n');
        if(QString::fromStdString(line).length()>0) {
            this->class_names.append(QString::fromStdString(line));
            this->num_classes++;
        }
    }
    std::cout<<"number of classes: "<<this->num_classes<<std::endl;
    this->output_num_array = output_size / (5 + this->num_classes);
}

void TensorRTEngine::set_conf_threshold(float thres) {
    this->conf_thres = thres;
}

void TensorRTEngine::set_nms_threshold(float thres) {
    this->nms_thres = thres;
}

QJsonObject TensorRTEngine::inference(cv::Mat image){

    if (!this->engine_loaded)
        return QJsonObject();
    if (image.empty())
        return QJsonObject();
    int img_w = image.cols;
    int img_h = image.rows;

    cv::Mat pr_img = static_resize(image, this->image_size);
    auto input_shape = pr_img.size();

    float* blob = blobFromImage(pr_img, this->pixel_range1);

    float scale = std::min(image_size.width() / (image.cols*1.0), image_size.height() / (image.rows*1.0));

//    const ICudaEngine& engine = context->getEngine();
    if (first) {
        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));
        this->first = false;
    }

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], blob, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(1, buffers, stream, nullptr);    //TensorRT 7.X.X
//    context->enqueueV2(buffers, stream, nullptr);     //TensorRT 8.X.X
    CHECK(cudaMemcpyAsync(prob, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    this->cuda_occupyed=true;

    std::vector<Object> objects;
    decode_outputs(prob, objects, scale, img_w, img_h, this->conf_thres, this->nms_thres, this->image_size, this->num_classes, this->output_num_array, this->obj_conf_enabled);
    delete blob;
    QJsonObject results;

//    std::cout<<"number of objects: "<<objects.size()<<std::endl;
    this->num_objects = objects.size();
    if (objects.size() > 0) {
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];
            QJsonObject this_result;

            this_result.insert("x", int(obj.rect.x));
            this_result.insert("y", int(obj.rect.y));
            this_result.insert("w", int(obj.rect.width));
            this_result.insert("h", int(obj.rect.height));
            this_result.insert("class_id", int(obj.label));
            this_result.insert("class", this->class_names.at(obj.label));
            this_result.insert("confidence", obj.prob);
            results.insert(QString::number(i), this_result);
//            std::cout<<"NO."<<i+1<<" ["<<int(obj.rect.x)<<","<<int(obj.rect.y)<<","<<int(obj.rect.width)<<","<<int(obj.rect.height)<<"], "<<this->class_names.at(obj.label).toStdString()<<" "<<obj.prob<<std::endl;
        }
    }

    return results;
}

void TensorRTEngine::release() {
    // Release stream and buffers
    if (this->engine_loaded){

//        std::cout<<1<<std::endl;

        cudaStreamDestroy(stream);
        if(this->cuda_occupyed) {
            CHECK(cudaFree(buffers[inputIndex]));
            CHECK(cudaFree(buffers[outputIndex]));
            this->cuda_occupyed=false;
        }

//        std::cout<<2<<std::endl;

        context->destroy();
        engine->destroy();
        runtime->destroy();

//        std::cout<<3<<std::endl;

//        delete runtime;
//        delete context;
//        delete engine;
        delete prob;
        this->engine_loaded=false;
    }
}

int TensorRTEngine::get_number_objects(){
    return this->num_objects;
}

QJsonObject readJson(QString json_file) {
    QFile loadFile(json_file);

    if(!loadFile.open(QIODevice::ReadOnly))
    {
        std::cout<<"could't open projects json"<<std::endl;
        return QJsonObject();
    }

    QByteArray allData = loadFile.readAll();
    loadFile.close();

    QJsonParseError json_error;
    QJsonDocument jsonDoc(QJsonDocument::fromJson(allData, &json_error));

    if(json_error.error != QJsonParseError::NoError)
    {
        std::cout<<"json error!"<<std::endl;
        return QJsonObject();
    }

    return jsonDoc.object();
}

void dumpJson(QJsonObject obj, QString file_name) {
    QFile file(file_name);
    if (!file.open(QIODevice::WriteOnly)) {
        std::cout<<"File open error"<<std::endl;
        return;
    }
    QJsonDocument jsonDoc;
    jsonDoc.setObject(obj);
    file.write(jsonDoc.toJson());
    file.close();
}

void TensorRTEngine::set_input_name(QString name) {
    this->input_name = name;
}

void TensorRTEngine::set_output_name(QString name) {
    this->output_name = name;
}

cv::Mat draw(cv::Mat image, QJsonObject objects, int draw_size=20, bool draw_label=true) {
    cv::Mat d_img = image.clone();

    cv::Scalar color;
    cv::Scalar txt_color;
    cv::Scalar txt_bk_color;
    cv::Size label_size;
    int baseLine = 0;
    int x, y, out_point_y;
    int line_thickness = std::round((double)draw_size / 10.0);

    for(int k=0; k<objects.keys().length(); k++){
        QJsonObject this_obj = objects.value(objects.keys().at(k)).toObject();
        color = get_color(this_obj.value("class_id").toInt());

        x = this_obj.value("x").toInt();
        y = this_obj.value("y").toInt();

        cv::rectangle(d_img,
                      cv::Rect(cv::Point(x, y),
                               cv::Size(this_obj.value("w").toInt(), this_obj.value("h").toInt())),
                      color,
                      line_thickness);

        if (draw_label){
            txt_color = (cv::mean(color)[0] > 127)?cv::Scalar(0, 0, 0):cv::Scalar(255, 255, 255);

            QString label = QString("%1 %2").arg(this_obj.value("class").toString()).arg(this_obj.value("confidence").toDouble(), 0, 'f', 2);

            label_size = cv::getTextSize(label.toStdString().c_str(), cv::LINE_AA, double(draw_size) / 30.0, (line_thickness>1)?line_thickness-1:1, &baseLine);

            txt_bk_color = color; // * 0.7;

            y = (y > d_img.rows)?d_img.rows:y + 1;

            out_point_y = y - label_size.height - baseLine;
            if (out_point_y >= 0) y = out_point_y;

            cv::rectangle(d_img, cv::Rect(cv::Point(x - (line_thickness - 1), y), cv::Size(label_size.width, label_size.height + baseLine)),
                          txt_bk_color, -1);

            cv::putText(d_img, label.toStdString(), cv::Point(x, y + label_size.height),
                        cv::LINE_AA, double(draw_size) / 30.0, txt_color, (line_thickness>1)?line_thickness-1:1);
        }

    }

    return d_img;
}


Detector::Detector(cv::VideoCapture cap, TensorRTEngine detector) {
    this->cap = cap;
    this->detector = detector;
    this->opened = cap.isOpened();

    this->t->start(1);
    this->t->moveToThread(timerThread);
    connect(this->t, SIGNAL(timeout()), this, SLOT(infer_and_show()), Qt::DirectConnection);

}

Detector::~Detector(){
    this->release();
}

void Detector::start(){
    timerThread->start();
    //std::cout<<t->isActive()<<std::endl;
}

bool Detector::isRunning(){
    return (t->isActive() && cap.isOpened());
}

void Detector::infer_and_show() {

    //std::cout<<"read image"<<cap.isOpened()<<std::endl;
    
    if (!cap.isOpened()) {
        opened = false;
        if (!this->loop)
            t->stop();
        return;
    }
    
    cap >> this->img;

    if (img.empty()) {
        opened = false;
        if (!this->loop)
            t->stop();
        else
            cap.release();
        return;
    }

    img_show = draw(img, detector.inference(img), 20, this->draw_label);
    updated = true;

}

void Detector::release() {
    this->cap.release();
    this->detector.release();
}
