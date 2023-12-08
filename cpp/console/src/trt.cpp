#include "image_utils/trt.h"


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
static Logger gLogger(ILogger::Severity::kERROR);   /* ILogger::Severity::kWARNING */


static void generate_yolo_proposals(int num_array, float* feat_blob, float prob_threshold, std::vector<detect::Object>& objects, int num_class)
{
    for (int anchor_idx = 0; anchor_idx < num_array; anchor_idx++)
    {
        const int basic_pos = anchor_idx * (num_class + 5);

        float box_objectness = feat_blob[basic_pos+4];    // obj conf
        if (box_objectness < prob_threshold) continue;

        int cls_idx=0;
        float max_prob= feat_blob[basic_pos + 5];
        for (int class_idx = 1; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
            if (box_cls_score > max_prob) {
                cls_idx = class_idx;
                max_prob = box_cls_score;
            }
        }

        max_prob *= box_objectness;
        if(max_prob > prob_threshold) {
            detect::Object obj;
            obj.rect.width = feat_blob[basic_pos+2];
            obj.rect.height = feat_blob[basic_pos+3];
            obj.rect.x = feat_blob[basic_pos] - obj.rect.width * 0.5f;
            obj.rect.y = feat_blob[basic_pos+1] - obj.rect.height * 0.5f;
            obj.label = cls_idx;
            obj.prob = max_prob;
            objects.push_back(obj);
        }
    } // point anchor loop
}

float* blobFromImage(cv::Mat& img, bool normalize=false){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    float range = normalize?255.0:1.0;
    for (size_t c = 0; int(c) < channels; c++)
        for (size_t  h = 0; int(h) < img_h; h++)
            for (size_t w = 0; int(w) < img_w; w++)
                blob[c * img_w * img_h + h * img_w + w] = (float)img.at<cv::Vec3b>(h, w)[c] / range;
    return blob;
}

static void decode_outputs(float* prob,
                           std::vector<detect::Object>& objects,
                           float scale,
                           const int img_w,
                           const int img_h,
                           float conf_thres,
                           float nms_thres,
                           cv::Size img_size,
                           int num_class,
                           int num_array) {
    
    std::vector<detect::Object> proposals;
    generate_yolo_proposals(num_array, prob,  conf_thres, proposals, num_class);
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

yoloNet::yoloNet() {
    cudaSetDevice(DEVICE);
    this->engine_loaded=false;
}

bool yoloNet::load_engine(std::string engine_name) {
    
    this->first = true;

    // if engine is loaded already, first release it and then load a new one.
    if (this->engine_loaded)
        this->release();
    this->engine_loaded = false;

    std::string cfg_file = engine_name.substr(0, engine_name.length() - 6) + "yaml";
    YAML::Node cfgs = YAML::LoadFile(cfg_file);
    cfg.INPUT_SIZE.height = cfgs["img_size"][0].as<int>();
    cfg.INPUT_SIZE.width = cfgs["img_size"][1].as<int>();
    cfg.INPUT_NAME = cfgs["input_name"].as<std::string>();
    cfg.OUTPUT_NAME = cfgs["output_name"].as<std::string>();
    cfg.NORMALIZE = cfgs["obj_conf_enabled"].as<bool>();
    cfg.NAMES = cfgs["names"].as<std::vector<std::string>>();
    cfg.loaded = true;


    char *trtModelStream{nullptr};
    ifstream file(engine_name, ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        this->size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    this->image_size.height = cfg.INPUT_SIZE.height;
    this->image_size.width = cfg.INPUT_SIZE.width;
    this->set_input_name(cfg.INPUT_NAME);
    this->set_output_name(cfg.OUTPUT_NAME);

    this->runtime = createInferRuntime(gLogger);
    assert(this->runtime != nullptr);
    this->engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    this->context = engine->createExecutionContext();
    assert(this->context != nullptr);
    delete[] trtModelStream;

    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex(cfg.INPUT_NAME.c_str());
    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    outputIndex = engine->getBindingIndex(cfg.OUTPUT_NAME.c_str());
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
            // cout<<"batch size of this engine file: "<<this->batch<<endl;

        }
        else this->output_size *= out_dims.d[j];
    }

    context->setBindingDimensions(inputIndex, out_dims);

    this->prob = new float[output_size];                //network output will be saved here
    this->num_classes = cfg.NAMES.size();
    this->output_num_array = output_size / (5 + this->num_classes);
    CHECK(cudaStreamCreate(&stream));
    // this->engine_loaded=true;
    /* not support batch > 1 now */
    this->engine_loaded = this->batch == 1;
    return this->engine_loaded;
}

void yoloNet::set_conf_threshold(float thres) {
    this->conf_thres = thres;
}

void yoloNet::set_nms_threshold(float thres) {
    this->nms_thres = thres;
}

std::vector<detect::Object> yoloNet::infer(cv::Mat image){

    std::vector<detect::Object> nullret;
    if (!this->engine_loaded)
        return nullret;
    if (image.empty())
        return nullret;
    
    int img_w = image.cols;
    int img_h = image.rows;

    // 1. pre-process
    cv::Mat pr_img = detect::static_resize(image, this->image_size);
    auto input_shape = pr_img.size();

    float* blob = blobFromImage(pr_img, this->normalize);

    float scale = std::min(image_size.width / (image.cols*1.0), image_size.height / (image.rows*1.0));

//    const ICudaEngine& engine = context->getEngine();
    if (first) {
        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));
        this->first = false;
    }

    // 2. inference
    CHECK(cudaMemcpyAsync(buffers[inputIndex], blob, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(1, buffers, stream, nullptr);    //TensorRT 7.X.X
    // context->enqueueV2(buffers, stream, nullptr);     //TensorRT 8.X.X
    CHECK(cudaMemcpyAsync(prob, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    this->cuda_occupyed=true;

    // 3. post-process
    std::vector<detect::Object> objects;
    decode_outputs(prob, objects, scale, img_w, img_h, this->conf_thres, this->nms_thres, this->image_size, this->num_classes, this->output_num_array);
    delete blob;
    
    return objects;
}

void yoloNet::release() {
    // Release stream and buffers
    if (this->engine_loaded){
        cudaStreamDestroy(stream);
        if(this->cuda_occupyed) {
            CHECK(cudaFree(buffers[inputIndex]));
            CHECK(cudaFree(buffers[outputIndex]));
            this->cuda_occupyed=false;
        }

        context->destroy();
        engine->destroy();
        runtime->destroy();
        delete prob;
        this->engine_loaded=false;
    }
}


void yoloNet::set_input_name(std::string name) {
    this->input_name = name;
}

void yoloNet::set_output_name(std::string name) {
    this->output_name = name;
}


