#ifndef MNN_H
#define MNN_H

#include <thread>

#include <opencv2/opencv.hpp>

#include "MNN/MNNDefine.h"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"
#include "MNN/AutoTime.hpp"
#include "MNN/Interpreter.hpp"

#include "image_utils/detect_process.h"
#include "yaml-cpp/yaml.h"

#include "print_utils.h"


namespace mnn_det {

    float img2mnn(cv::Mat &img, MNN::Tensor *input_tensor, int batch_id=0) {
        assert(batch_id < input_tensor->shape().at(0));
        assert(input_tensor->shape().size() == 4);

        cv::Size fixed_size(input_tensor->shape().at(3), input_tensor->shape().at(2));


        // std::cout << fixed_size << std::endl;


        auto resize_info = detect::resizeAndPad(img, fixed_size, false, false);

        int img_C3 = resize_info.resized_img.channels();
        auto nchwTensor = new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE);

        int fixedlw = fixed_size.height * fixed_size.width;
        int fixedchw = img_C3 * fixedlw;
        for (int i = 0; i < fixed_size.height; i++)
            for (int j = 0; j < fixed_size.width; j++)
                for (int k = 0; k < img_C3; k++) {
                    nchwTensor->host<float>()[batch_id * fixedchw + k * fixedlw + i * fixed_size.width + j] = resize_info.resized_img.at<cv::Vec3b>(i, j)[k];
                }

        input_tensor->copyFromHostTensor(nchwTensor);
        delete nchwTensor;

        return resize_info.factor;
    }

    class YOLO {

        std::shared_ptr<MNN::Interpreter> net; //模型翻译创建
        MNN::ScheduleConfig config;            //计划配置
        MNN::Session *session;

        MNN::Tensor *inputTensor;
        MNN::Tensor *outputTensor;

        cv::Size input_size, img_size;
        int num_batch;
        int num_dets;
        int length_array;

        float *preds;

        std::vector<float> batch_ratios;
        std::vector<std::vector<detect::Object>> batch_objects;

    public:
        std::string mnn_model;
        std::vector<std::string> names;
        int num_threads=4;

        float conf_thres=0.25;
        float nms_thres=0.45;

        bool config_set=false;
        bool model_loaded=false;

        cv::Mat *current_img;
        float current_ratio;
        MNNForwardType DEVICE=MNN_FORWARD_AUTO;

        YOLO() {};

        ~YOLO() {
            INFO << "release yolo" << ENDL;
            // delete session;
            // delete inputTensor;
            // delete outputTensor;
            // delete current_img;
            // delete preds;
        }

        YOLO(std::string mnn_config) {
            set_config(mnn_config);
            // INFO << config_set << ENDL;
        }

        void set_config(std::string mnn_config) {
            YAML::Node cfg = YAML::LoadFile(mnn_config);
            mnn_model = cfg["model"].as<std::string>();
            names = cfg["names"].as<std::vector<std::string>>();
            num_threads = cfg["num_threads"].as<int>();
            conf_thres = cfg["conf_thres"].as<float>();
            nms_thres = cfg["nms_thres"].as<float>();
            config_set = true;
        }

        void set_device(bool _cpu=false, bool _gpu=false) {
            if (_cpu) DEVICE = MNN_FORWARD_CPU;
            if (_gpu) DEVICE = MNN_FORWARD_CUDA;
        }

        void set_confidence_threshold(float _value) {
            conf_thres = _value;
        }

        void set_nms_threshold(float _value) {
            nms_thres = _value;
        }

        bool load_model() {
            if (!config_set) return false;

            net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(this->mnn_model.c_str()));
            config.numThread = num_threads;

            // config.type = MNN_FORWARD_AUTO;
            config.type = DEVICE;

            session = net->createSession(config);
            //获取输入输出tensor
            inputTensor = net->getSessionInput(session, NULL);
            outputTensor = net->getSessionOutput(session, NULL);

            length_array = outputTensor->shape().at(2);

            assert(inputTensor->shape().size() == 4);    // batch, c, h, w
            assert(outputTensor->shape().size() == 3);   // batch, num_dets, array
            assert(length_array == names.size()+5);
            
            num_batch = inputTensor->shape().at(0);
            input_size.height = inputTensor->shape().at(2);
            input_size.width = inputTensor->shape().at(3);
            num_dets = outputTensor->shape().at(1);

            model_loaded=true;

            return model_loaded;
        }

        void _forward(cv::Mat *img, int batch_id=0) {
            current_img = img;
            img_size = current_img->size();
            current_ratio = img2mnn(*current_img, inputTensor, batch_id);
            net->runSession(session);

            auto outNchwTensor = new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE);
            outputTensor->copyToHostTensor(outNchwTensor);
            preds = outNchwTensor->host<float>();
            // INFO << "RATIO: " << current_ratio << ENDL;
        }

        void __preprocess_batch_once(cv::Mat img, int batch_id=0) {
            img_size = img.size();
            batch_ratios[batch_id] = img2mnn(img, inputTensor, batch_id);
        }

        void _forward_multi(std::vector<cv::Mat> imgs) {
            assert(imgs.size()==num_batch);
            batch_ratios.clear();
            batch_ratios.resize(num_batch);

            std::vector<std::thread> threads;
            for (int i=0;i<num_batch;i++) {
                std::thread _t(std::mem_fn(&YOLO::__preprocess_batch_once), this, imgs.at(i), i);
                threads.push_back(std::move(_t));
            }
            for (auto& _t: threads) _t.join();

            net->runSession(session);

            auto outNchwTensor = new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE);
            outputTensor->copyToHostTensor(outNchwTensor);
            preds = outNchwTensor->host<float>();

            INFO << preds << ENDL;
        }

        void generate_yolo_proposals(std::vector<detect::Object>& objects, int batch_id) {
            for (int anchor_idx = 0; anchor_idx < num_dets; anchor_idx++) {
                const int basic_pos = batch_id * num_dets + anchor_idx * length_array;
                
                // 解析类别及其置信度
                int label = -1;
                float prob = 0.0;
                // INFO << 1 << ENDL;
                float box_objectness = preds[basic_pos+4];    // obj conf
                // INFO << 2 << ENDL;
                for (int class_idx = 0; class_idx < length_array - 5; class_idx++)
                {
                    float box_cls_score = preds[basic_pos + 5 + class_idx];
                    float box_prob = box_objectness * box_cls_score;
                    if (box_prob > conf_thres && box_prob > prob) {
                        label = class_idx;
                        prob = box_prob;
                    }
                }
                // INFO << 3 << ENDL;

                // 若置信度大于阈值则输出
                if(prob > conf_thres) {
                    detect::Object obj;
                    obj.rect.width = preds[basic_pos+2];
                    obj.rect.height = preds[basic_pos+3];
                    obj.rect.x = preds[basic_pos+0] - obj.rect.width * 0.5f;
                    obj.rect.y = preds[basic_pos+1] - obj.rect.height * 0.5f;
                    
                    obj.label = label;
                    obj.prob = prob;

                    objects.push_back(obj);
                }
                // INFO << 4 << ENDL;
            }
        }

        void _decode_outputs(int batch_id=0) {
            batch_objects[batch_id] = this->_decode_output(batch_id, batch_ratios[batch_id]);
        }

        std::vector<detect::Object> _decode_output(int batch_id=0, float ratio=1.0) {
            std::vector<detect::Object> proposals, objects;
            std::vector<int> picked;

            this->generate_yolo_proposals(proposals, batch_id);
            detect::qsort_descent_inplace(proposals);
            detect::nms_sorted_bboxes(proposals, picked, nms_thres);

            int count = picked.size();
            objects.resize(count);
            for (int i = 0; i < count; i++)
            {
                objects[i] = proposals[picked[i]];

                float x0 = (objects[i].rect.x) * ratio;
                float y0 = (objects[i].rect.y) * ratio;
                float x1 = (objects[i].rect.x + objects[i].rect.width) * ratio;
                float y1 = (objects[i].rect.y + objects[i].rect.height) * ratio;

                // clip
                x0 = std::max(std::min(x0, (float)(img_size.width - 1)), 0.f);
                y0 = std::max(std::min(y0, (float)(img_size.height - 1)), 0.f);
                x1 = std::max(std::min(x1, (float)(img_size.width - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(img_size.height - 1)), 0.f);

                objects[i].rect.x = x0;
                objects[i].rect.y = y0;
                objects[i].rect.width = x1 - x0;
                objects[i].rect.height = y1 - y0;
            }

            return objects;

        }

        std::vector<std::vector<detect::Object>> batch_infer(std::vector<cv::Mat> images) {
            this->_forward_multi(images);
            batch_objects.clear();
            batch_objects.resize(num_batch);
            std::vector<std::thread> threads;
            for (int i=0;i<num_batch;i++) {
                std::thread _t(std::mem_fn(&YOLO::_decode_outputs), this, i);
                threads.push_back(std::move(_t));
            }
            for (auto& _t: threads) _t.join();

            return batch_objects;
        }

        std::vector<detect::Object> infer(cv::Mat image) {
            this->_forward(&image, 0);
            return this->_decode_output(0, current_ratio);
        }


    };

}


// void 

#endif
#define MNN_H