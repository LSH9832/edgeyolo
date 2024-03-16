/*
 * yolo.h
 *
 *  Created on: Mar 6, 2024
 *      Author: LSH9832
 */

#ifndef INCLUDE_DETECT_MDC_YOLO_H_
#define INCLUDE_DETECT_MDC_YOLO_H_

#include <yaml-cpp/yaml.h>

#include "detect/mdc/dvpp_handler.h"
#include "detect/mdc/davinci_net.h"
#include "detect/common.h"
#include "print_utils.h"

#include <fstream>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>




namespace MDC_DET {

    class YOLO {

    public:
    	DVPPHandler dvpp;
		DavinciNet davinciNet;

		std::shared_ptr<Blob<float>> data;
		std::shared_ptr<Blob<float16_t>> data_fp16;
		std::shared_ptr<Blob<float>> preds;
		std::shared_ptr<Blob<float16_t>> preds_fp16;

		std::vector<std::shared_ptr<Blob<float>>> sp_preds;

    	int batch_size=1;
    	int length_array = 85;
    	std::vector<int> img_size = {640, 640};
    	cv::Size imgSize, ori_size;
		std::string input_name = "image";
		std::string output_name = "preds";
		std::vector<std::string> output_names;
		std::vector<int> gridsX, gridsY;
		std::vector<int> strides = {8, 16, 32};
		std::vector<int> rerank = {0, 1, 2, 3, 4, 5, 6, 7, 8};
		std::vector<std::string> names;
		std::string model_path;

		float current_ratio = 1.;

		double conf_thres=0.25, nms_thres=0.5;
		double us_ct = detect::unsigmoid(0.25);

		bool init = false, model_loaded=false, debug_mode=false, seperated_head=false;

		int num_dets = 0;

		TimeCount t;

    	YOLO() {}

    	YOLO(std::string cfg_path, bool debug=false) {
    		debug_mode = debug;
    		load_cfg(cfg_path, debug);
    		if (init) {
    			load_model();
    		}
    	}

    	void setConfThreshold(double _value) {
    		this->conf_thres = _value;
    	}

    	void setNMSThreshold(double _value) {
    		this->nms_thres = _value;
    	}

    	std::vector<detect::Object> infer(cv::Mat &img) {
    		// check
    		std::vector<detect::Object> objs;
    		if (!model_loaded) {
    			std::cerr << "Davinci Net not init!" << std::endl;
    			return objs;
    		}

    		// pre-process

    		t.tic(0);
    		if (debug_mode) std::cout << "1. start pre-process" << std::endl;
    		if (debug_mode) std::cout << "1.1 resize and pad" << std::endl;

    		ori_size = img.size();

    		detect::resizeInfo info;
//    		info = detect::resizeAndPad(img, imgSize);
    		detect::static_resize(img, imgSize, info);
    		current_ratio = info.factor;

    		int img_C3 = info.resized_img.channels();
			int fixedlw = imgSize.height * imgSize.width;
			int fixedchw = img_C3 * fixedlw;

			if (debug_mode) std::cout << "1.2 fill image data to valuable" << std::endl;

			float* _d = data->Data();
			for (int k = 0; k < img_C3; k++) {
				int klw = k * fixedlw;
				for (int i = 0; i < imgSize.height; i++) {
					int iw = i * imgSize.width;
					for (int j = 0; j < imgSize.width; j++) {
						_d[klw + iw + j] = static_cast<float>(info.resized_img.at<cv::Vec3b>(i, j)[k]);
					}
				}
			}
			t.toc(0);

			// inference
			t.tic(1);
			if (debug_mode) std::cout << "2. start davinciNet inference" << std::endl;
			int ret = davinciNet.Inference();
			if (ret != 0) {
				if (debug_mode) std::cerr << "Davinci Net inference failed!" << std::endl;
				return objs;
			}
			t.toc(1);

			// post-process
			t.tic(2);
			if (debug_mode) std::cout << "3. start post-process" << std::endl;
			objs = this->_decode_output();
			t.toc(2);
			return objs;
    	}

    	void generate_yolo_proposals(std::vector<detect::Object>& objects, int batch_id) {
    		if (debug_mode) std::cout << "3.1.1 get predict data" << std::endl;

			float* _p = this->preds->Data();

			if (debug_mode) std::cout << "3.1.2 for each grid cell" << std::endl;
			for (int anchor_idx = 0; anchor_idx < num_dets; anchor_idx++) {

				if (debug_mode) if (anchor_idx == 0) std::cout << "3.1.2.1 get position of result of current grid cell" << std::endl;

				const int basic_pos = batch_id * num_dets + anchor_idx * length_array;

				// 解析类别及其置信度
				int label = -1;
				float prob = 0.0;

				if (debug_mode) if (anchor_idx == 0) std::cout << "3.1.2.2 get obj conf" << std::endl;

				float box_objectness = _p[basic_pos+4];    // obj conf

				if (debug_mode) if (anchor_idx == 0) std::cout << "3.1.2.3 get maximum cls conf and its cls id" << std::endl;
				for (int class_idx = 0; class_idx < length_array - 5; class_idx++)
				{
					float box_cls_score = _p[basic_pos + 5 + class_idx];
					float box_prob = box_objectness * box_cls_score;
					if (box_prob > conf_thres && box_prob > prob) {
						label = class_idx;
						prob = box_prob;
					}
				}

				if (debug_mode) if (anchor_idx == 0) std::cout << "3.1.2.4 reserve item if conf > threshold" << std::endl;
				// 若置信度大于阈值则输出
				if(prob > conf_thres) {
					detect::Object obj;
					obj.rect.width = _p[basic_pos+2];
					obj.rect.height = _p[basic_pos+3];
					obj.rect.x = _p[basic_pos+0] - obj.rect.width * 0.5f;
					obj.rect.y = _p[basic_pos+1] - obj.rect.height * 0.5f;

					obj.label = label;
					obj.prob = prob;

					objects.push_back(obj);
				}
			}

			if (debug_mode) std::cout << "3.1.2.5 end generating proposals" << std::endl;
		}

    	void decode_head(std::vector<detect::Object>& objects, float* reg, float* obj, float* cls,
    			         int x_grid, int y_grid, int stride=8, int batch_id=0) {
    		/**
    		 * decode seperated head
    		 * box, obj_conf, cls_conf
    		 */
    		int num_grids = x_grid * y_grid;
    		int offset, bias, cls_batch_bias, max_id;
    		float obj_prob, max_class_prob, prob;

			for (int i=0;i<y_grid;i++) {
				offset = i * x_grid;
				for(int j=0;j<x_grid;j++) {
					bias = offset + j;
					obj_prob = obj[batch_id * num_grids + bias];

					if (obj_prob < us_ct) continue;

					cls_batch_bias = batch_id * length_array * num_grids;
					max_class_prob = cls[cls_batch_bias + bias];
					max_id = 0;
					prob = 0.;
					for (int k = 1; k < length_array; ++k) {
						prob = cls[cls_batch_bias + k * num_grids + bias];
						if (prob > max_class_prob) {
							max_id = k;
							max_class_prob = prob;
						}
					}

					prob = detect::sigmoid(obj_prob) * detect::sigmoid(max_class_prob);

					if (prob > conf_thres) {
						detect::Object this_obj;

						int reg_batch_bias = batch_id * 4 * num_grids;
						this_obj.label = max_id;
						this_obj.prob = prob;

						this_obj.rect.width = expf(reg[reg_batch_bias + 2 * num_grids + bias]) * stride;
						this_obj.rect.height = expf(reg[reg_batch_bias + 3 * num_grids + bias]) * stride;

						this_obj.rect.x = (reg[reg_batch_bias + 0 * num_grids + bias] + j) * stride - 0.5f * this_obj.rect.width;
						this_obj.rect.y = (reg[reg_batch_bias + 1 * num_grids + bias] + i) * stride - 0.5f * this_obj.rect.height;

						objects.push_back(this_obj);
					}
				}
			}

    	}

		std::vector<detect::Object> _decode_output(int batch_id=0) {
			std::vector<detect::Object> proposals, objects;
			std::vector<int> picked;

			if (debug_mode) std::cout << "3.1 generate yolo proposals" << std::endl;

			if (seperated_head) {
				for (int i=0;i<3;i++) {
					this->decode_head(proposals, sp_preds[i]->Data(), sp_preds[i+3]->Data(), sp_preds[i+6]->Data(),
							          gridsX[i], gridsY[i], strides[i], batch_id);
				}
			}
			else this->generate_yolo_proposals(proposals, batch_id);

			if (debug_mode) std::cout << "3.2 qsort and nms" << std::endl;
			detect::qsort_descent_inplace(proposals);
			detect::nms_sorted_bboxes(proposals, picked, nms_thres);

			if (debug_mode) std::cout << "3.3 pick reserved results and return" << std::endl;
			int count = picked.size();
			objects.resize(count);
			for (int i = 0; i < count; i++)
			{
				objects[i] = proposals[picked[i]];

				float x0 = (objects[i].rect.x) * current_ratio;
				float y0 = (objects[i].rect.y) * current_ratio;
				float x1 = (objects[i].rect.x + objects[i].rect.width) * current_ratio;
				float y1 = (objects[i].rect.y + objects[i].rect.height) * current_ratio;

				// clip
				x0 = std::max(std::min((float)x0, (float)(ori_size.width - 1)), 0.f);
				y0 = std::max(std::min((float)y0, (float)(ori_size.height - 1)), 0.f);
				x1 = std::max(std::min((float)x1, (float)(ori_size.width - 1)), 0.f);
				y1 = std::max(std::min((float)y1, (float)(ori_size.height - 1)), 0.f);

				objects[i].rect.x = x0;
				objects[i].rect.y = y0;
				objects[i].rect.width = x1 - x0;
				objects[i].rect.height = y1 - y0;
			}

			return objects;
		}

    	int load_model() {
    		if (!init) return -1;

    		aclError ret = aclInit(nullptr);
			if (ret != ACL_ERROR_NONE) {
				std::cerr << "acl init failed!" << std::endl;
				return -1;
			}

			ret = aclrtSetDevice(0);
			if (ret != ACL_ERROR_NONE) {
				std::cerr << "acl set device failed!" << std::endl;
				return -1;
			}

			ret = davinciNet.Init(model_path);
			if (ret != 0) {
				std::cerr << "Davinci Net init failed!" << std::endl;
				return 1;
			}

			input_name = davinciNet.input_names[0];
			batch_size = davinciNet.inputs_dims[0][0];
			imgSize.height = davinciNet.inputs_dims[0][2];
			imgSize.width = davinciNet.inputs_dims[0][3];

			seperated_head = davinciNet.output_names.size() > 1;
			if (seperated_head) {
				output_names.resize(9);
				assert(davinciNet.output_names.size()==9);
				for (int i=0;i<9;i++) {
					output_names[i] = davinciNet.output_names[rerank[i]];
				}
				length_array = davinciNet.outputs_dims[6][1];
				for (int i=names.size(); i<length_array; i++) {
					names.push_back("unknown_" + std::to_string(i+1));
				}
			}
			else {
				output_name = davinciNet.output_names[0];
				num_dets = davinciNet.outputs_dims[0][1];
				length_array = davinciNet.outputs_dims[0][2];
			}
			data = std::static_pointer_cast<Blob<float>>(davinciNet.GetBlob(input_name));
			if (seperated_head) {
			    sp_preds.resize(9);
				for(int i=0; i<output_names.size(); i++) {
					sp_preds[i] = std::static_pointer_cast<Blob<float>>(davinciNet.GetBlob(output_names[i]));
					if (i > 5) {
						gridsX.push_back(davinciNet.outputs_dims[rerank[i]][3]);
						gridsY.push_back(davinciNet.outputs_dims[rerank[i]][2]);
					}
				}
//				for (int i=0;i<3;i++) {
//					gridsX.push_back()
//				}
			}
			else {
				preds = std::static_pointer_cast<Blob<float>>(davinciNet.GetBlob(output_name));
			}

    		model_loaded = true;
    		return 0;
    	}

    	void load_cfg(std::string cfg_path, bool debug=false) {
    		YAML::Node cfg = YAML::LoadFile(cfg_path);

			std::vector<std::string> keys = {"names", "model_path", "rerank"};

			for(int idx=0;idx<keys.size();idx++) {
				auto node = cfg[keys[idx]];
				if (node.IsDefined()) {
					switch (idx) {
					case 0:
						names = node.as<std::vector<std::string>>();
						break;
					case 1:
						model_path = node.as<std::string>();
						init=true;
						break;
					case 2:
						rerank = node.as<std::vector<int>>();
						assert(rerank.size()==9);
						break;
					default:
						std::cout << "error key index!" << std::endl;
						return;
					}
				}
//				else WARN << "" << ENDL;
			}

    	}


    };
}



#endif /* INCLUDE_DETECT_MDC_YOLO_H_ */
