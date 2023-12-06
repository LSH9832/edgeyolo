#ifndef RKNN_H
#define RKNN_H

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#define _BASETSD_H

#include "RgaUtils.h"
#include "rknn_api.h"
#include "image_utils/detect_process.h"

#include <yaml-cpp/yaml.h>


namespace RKNN {

    static void dump_tensor_attr(rknn_tensor_attr *attr) {
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i) shape_str += ", " + std::to_string(attr->dims[i]);
    
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
            "type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
            attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
    }

    static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) {
        unsigned char *data;
        int ret;
        data = NULL;
        if (NULL == fp) return NULL;

        ret = fseek(fp, ofst, SEEK_SET);
        if (ret != 0) {
            printf("blob seek failure.\n");
            return NULL;
        }

        data = (unsigned char *)malloc(sz);
        if (data == NULL) {
            printf("buffer malloc failure.\n");
            return NULL;
        }
        ret = fread(data, 1, sz, fp);
        return data;
    }

    struct RUNING_MODE {
        rknn_core_mask CORE_0 = RKNN_NPU_CORE_0;
        rknn_core_mask CORE_1 = RKNN_NPU_CORE_1;
        rknn_core_mask CORE_2 = RKNN_NPU_CORE_2;
        rknn_core_mask CORE_0_1 = RKNN_NPU_CORE_0_1;
        rknn_core_mask CORE_0_1_2 = RKNN_NPU_CORE_0_1_2;
    };

    inline static int32_t __clip(float val, float min, float max) {
        float f = val <= min ? min : (val >= max ? max : val);
        return f;
    }

    static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
        float dst_val = (f32 / scale) + zp;
        int8_t res = (int8_t)__clip(dst_val, -128, 127);
        return res;
    }

    static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { 
        return ((float)qnt - (float)zp) * scale; 
    }

    float sigmoid(float x) {
        return 1 / (1 + expf(-x));
    }

    double unsigmoid(double x) {
        return -logf((1 - x) / x);
    }

    class YOLO {

        YAML::Node cfg;
        rknn_context ctx;
        
        std::string model_file;
        cv::Size input_size;
        bool model_loaded=false;
        rknn_core_mask mode=RKNN_NPU_CORE_0_1_2;

        rknn_input inputs[1];
        rknn_output outputs[9];

        int num_classes=80;

        double conf_thres=0.25;
        double nms_thres=0.45;

        int strides[3] = {8, 16, 32};

        std::vector<int8_t> conf_thres_int8s;
        std::vector<int> all_num_dets;
        std::vector<int> all_x_grids;
        std::vector<int> all_y_grids;

        std::vector<int32_t> output_zps;
        std::vector<float> output_scales;

    public:
        std::vector<std::string> names;

        YOLO() {}

        YOLO(std::string weight_file, double _conf_thres=0.25, double _nms_thres=0.45) {
            model_file = weight_file;
            std::string cfg_file = weight_file;
            for (int i=0;i<4;i++) cfg_file.pop_back();
            cfg_file += "yaml";
            cfg = YAML::LoadFile(cfg_file);
            names = cfg["names"].as<std::vector<std::string>>();
            input_size.height = cfg["img_size"][0].as<int>();
            input_size.width = cfg["img_size"][1].as<int>();
            conf_thres = _conf_thres;
            nms_thres = _nms_thres;
        }

        void set_running_core_mode(rknn_core_mask _core_mask) {
            mode = _core_mask;
        }

        int load_model() {
            int ret = 0;
            const char* filename = model_file.c_str();
            FILE *fp;
            unsigned char *data;

            fp = fopen(filename, "rb");
            if (NULL == fp) {
                printf("Open file %s failed.\n", filename);
                return -1;
            }

            fseek(fp, 0, SEEK_END);
            int size = ftell(fp);

            data = load_data(fp, 0, size);

            fclose(fp);

            ret = rknn_init(&ctx, data, size, 0, NULL);
            if (ret < 0) {
                printf("rknn_init error ret=%d\n", ret);
                return ret;
            }

            ret = rknn_set_core_mask(ctx, mode);
            if (ret < 0) {
                printf("rknn_init core error ret=%d\n", ret);
                return ret;
            }

            rknn_sdk_version version;
            ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
            if (ret < 0) {
                printf("rknn_init error ret=%d\n", ret);
                return -1;
            }
            printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

            rknn_input_output_num io_num;
            ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
            if (ret < 0) {
                printf("rknn_init error ret=%d\n", ret);
                return -1;
            }
            printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

            rknn_tensor_attr input_attrs[io_num.n_input];
            memset(input_attrs, 0, sizeof(input_attrs));
            for (int i = 0; i < io_num.n_input; i++) {
                input_attrs[i].index = i;
                ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
                if (ret < 0)
                {
                    printf("rknn_init error ret=%d\n", ret);
                    return -1;
                }
                dump_tensor_attr(&(input_attrs[i]));
            }

            rknn_tensor_attr output_attrs[io_num.n_output];
            memset(output_attrs, 0, sizeof(output_attrs));
            assert(io_num.n_output == 9);
            for (int i = 0; i < io_num.n_output; i++) {
                output_attrs[i].index = i;
                ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
                dump_tensor_attr(&(output_attrs[i]));
                output_zps.push_back(output_attrs[i].zp);
                output_scales.push_back(output_attrs[i].scale);
                if (i>5) {
                    all_num_dets.push_back(output_attrs[i].dims[2] * output_attrs[i].dims[3]);
                    all_x_grids.push_back(output_attrs[i].dims[3]);
                    all_y_grids.push_back(output_attrs[i].dims[2]);
                }
                else if (i>2) {
                    conf_thres_int8s.push_back(qnt_f32_to_affine(unsigmoid(conf_thres), output_attrs[i].zp, output_attrs[i].scale));
                }
            }
            num_classes = output_attrs[8].dims[1];
            if (num_classes != names.size()) {
                printf("number of classes in config does not match the real output!");
                if (names.size() < num_classes) {
                    for (int i=names.size();i<=num_classes;i++) names.push_back("unknown_" + std::to_string(i));
                }
            }

            int channel = 3;
            int width = 0;
            int height = 0;
            if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
                printf("model is NCHW input fmt\n");
                channel = input_attrs[0].dims[1];
                height = input_attrs[0].dims[2];
                width = input_attrs[0].dims[3];
            }
            else {
                printf("model is NHWC input fmt\n");
                height = input_attrs[0].dims[1];
                width = input_attrs[0].dims[2];
                channel = input_attrs[0].dims[3];
            }
            if (input_size.width != width) {
                printf("config size width %d not match model input width %d, modify to correct value", 
                       input_size.width, width);
                input_size.width = width;
            }
            if (input_size.height != height) {
                printf("config size height %d not match model input height %d, modify to correct value", 
                       input_size.height, height);
                input_size.height = height;
            }

            printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

            memset(inputs, 0, sizeof(inputs));
            inputs[0].index = 0;
            inputs[0].type = RKNN_TENSOR_UINT8;
            inputs[0].size = width * height * channel;
            inputs[0].fmt = RKNN_TENSOR_NHWC;
            inputs[0].pass_through = 0;

            memset(outputs, 0, sizeof(outputs));
            for (int i=0;i<9;i++) outputs[i].want_float = 0;

            model_loaded = true;
            return 0;
        }

        double _forward(cv::Mat _img) {
            detect::resizeInfo resize_info = detect::resizeAndPad(_img, input_size, false, false);
            int ret = 0;
            inputs[0].buf = resize_info.resized_img.data;
            rknn_inputs_set(ctx, 1, inputs);
            ret = rknn_run(ctx, NULL);
            ret = rknn_outputs_get(ctx, 9, outputs, NULL);
            return resize_info.factor;
        }

        void conf_filter(int8_t *reg, int8_t *obj_conf, int8_t *cls_conf, int x_grid, int y_grid, int stride,
                         int8_t conf_thres_i8, std::vector<detect::Object>& objs,
                         int32_t reg_zp, int32_t obj_zp, int32_t cls_zp, 
                         float reg_scale, float obj_scale, float cls_scale) {
            
            int num_grids = x_grid * y_grid;
            for (int i=0;i<y_grid;i++) {
                int offset = i * x_grid;
                for(int j=0;j<x_grid;j++) {
                    int bias = offset + j;
                    int8_t obj_prob = obj_conf[0 * num_grids + bias];

                    if (obj_prob < conf_thres_i8) continue;

                    int8_t max_class_prob = cls_conf[0 * num_grids + bias];
                    int max_id = 0;
                    for (int k = 1; k < num_classes; ++k) {
                        int8_t prob = cls_conf[k * num_grids + bias];
                        if (prob > max_class_prob) {
                            max_id = k;
                            max_class_prob = prob;
                        }
                    }

                    float prob = sigmoid(deqnt_affine_to_f32(obj_prob, obj_zp, obj_scale)) * 
                                 sigmoid(deqnt_affine_to_f32(max_class_prob, cls_zp, cls_scale));

                    if (prob > conf_thres) {
                        detect::Object this_obj;
                        this_obj.label = max_id;
                        this_obj.prob = prob;
                        this_obj.rect.x = deqnt_affine_to_f32(reg[0 * num_grids + bias], reg_zp, reg_scale);
                        this_obj.rect.y = deqnt_affine_to_f32(reg[1 * num_grids + bias], reg_zp, reg_scale);
                        this_obj.rect.width = deqnt_affine_to_f32(reg[2 * num_grids + bias], reg_zp, reg_scale);
                        this_obj.rect.height = deqnt_affine_to_f32(reg[3 * num_grids + bias], reg_zp, reg_scale);

                        this_obj.rect.x = (this_obj.rect.x + j) * stride;
                        this_obj.rect.y = (this_obj.rect.y + i) * stride;
                        this_obj.rect.width = expf(this_obj.rect.width) * stride;
                        this_obj.rect.height = expf(this_obj.rect.height) * stride;

                        this_obj.rect.x -= 0.5f * this_obj.rect.width;
                        this_obj.rect.y -= 0.5f * this_obj.rect.height;

                        objs.push_back(this_obj);
                    }
                }
            }
                
        }

        void post_process(double factor, 
                          std::vector<detect::Object>& objects, 
                          cv::Size ori_img_size) {
            std::vector<detect::Object> obj_rests;
            std::vector<int> picked;

            // conf_filter(output, obj_rests);
            for (int i=0;i<3;i++) {
                conf_filter((int8_t *)outputs[i].buf, (int8_t *)outputs[i+3].buf, (int8_t *)outputs[i+6].buf,
                            all_x_grids[i], all_y_grids[i], strides[i], conf_thres_int8s[i], obj_rests,
                            output_zps[i], output_zps[i+3], output_zps[i+6],
                            output_scales[i], output_scales[i+3], output_scales[i+6]);
            }

            detect::qsort_descent_inplace(obj_rests);
            detect::nms_sorted_bboxes(obj_rests, picked, nms_thres);

            int count = picked.size();
            objects.resize(count);
            for (int i = 0; i < count; i++)
            {
                objects[i] = obj_rests[picked[i]];

                float x0 = (objects[i].rect.x) * factor;
                float y0 = (objects[i].rect.y) * factor;
                float x1 = (objects[i].rect.x + objects[i].rect.width) * factor;
                float y1 = (objects[i].rect.y + objects[i].rect.height) * factor;

                // clip
                x0 = std::max(std::min(x0, (float)(ori_img_size.width - 1)), 0.f);
                y0 = std::max(std::min(y0, (float)(ori_img_size.height - 1)), 0.f);
                x1 = std::max(std::min(x1, (float)(ori_img_size.width - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(ori_img_size.height - 1)), 0.f);

                objects[i].rect.x = x0;
                objects[i].rect.y = y0;
                objects[i].rect.width = x1 - x0;
                objects[i].rect.height = y1 - y0;
            }
        }

        std::vector<detect::Object> infer(cv::Mat _img) {
            std::vector<detect::Object> objs;
            if (!model_loaded) return objs;

            double factor = this->_forward(_img);

            post_process(factor, objs, _img.size());
            return objs;
        }
    };
}




#endif
#define RKNN_H
