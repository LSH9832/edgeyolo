/*
 * main.h
 *
 *  Created on: Mar 6, 2024
 *      Author: LSH9832
 */
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "argparser.h"
#include "print_utils.h"
#include "detect/mdc/yolo.h"
#include "cpp-py/str.h"


argsutil::argparser get_args(int argc, char** argv) {
	argsutil::argparser parser("MDC edgeyolo infer parser");

	parser.add_option<std::string>("-c", "--cfg", "model config file", "../models/cfg.yaml");
	parser.add_option<std::string>("-s", "--source", "image source", "~/Videos/test.avi");
	parser.add_option<std::string>("-o", "--output", "result output file, image is default result.jpg", "");
	parser.add_option<double>("-ct", "--conf-thres", "confidence threshold", 0.25);
	parser.add_option<double>("-nt", "--nms-thres", "nms threshold", 0.5);
	parser.add_option<bool>("-v", "--view-only", "view input and output shape only", false);
	parser.add_option<bool>("-d", "--debug", "debug mode", false);
	parser.add_option<int>("-r", "--repeat", "repeat inference for one image for testing delay", 1);

	parser.add_help_option();
	parser.parse(argc, argv);

	return parser;
}


int main(int argc, char** argv) {

	auto parser = get_args(argc, argv);

	std::string cfg_name = parser.get_option<std::string>("--cfg");
	std::string source = parser.get_option<std::string>("--source");
	std::string output = parser.get_option<std::string>("--output");
	bool debug = parser.get_option<bool>("--debug");

	double ct = parser.get_option<double>("--conf-thres");
	double nt = parser.get_option<double>("--nms-thres");

	if (debug) std::cout << "init yolo net" << std::endl;
	auto yoloNet = MDC_DET::YOLO(cfg_name, debug);
	if (debug) std::cout << "end init yolo net" << std::endl;

	if (parser.get_option<bool>("--view-only")) return 0;

	yoloNet.setConfThreshold(ct);
	yoloNet.setNMSThreshold(nt);
	detect::names = yoloNet.names;

	cv::Mat _img;
	TimeCount tc;
	std::vector<detect::Object> objs;

	bool source_img=false;
	for(std::string suffix: {".jpg", ".png", ".bmp", ".jpeg", ".webp",
		                     ".JPG", ".PNG", ".BMP", ".JPEG", ".WEBP"})
	{
		if (pyStr::endswith(source, suffix)) {
			INFO << "source ends with " << suffix << ENDL;
			source_img = true;
			break;
		}
	}

	if (source_img) {
		if (!output.length()) output = "output.jpg";
		_img = cv::imread(source);
		if (!_img.empty()) {

			if (debug) std::cout << "start infer" << std::endl;
			int repeat_time = parser.get_option<int>("--repeat");


			while (repeat_time--) {
				tc.tic(0);
				objs = yoloNet.infer(_img);
				tc.toc(0);
				std::cout << "end infer. num objects: " << objs.size() << ", time total:" << tc.get_timeval_f(0) << " ms, pre: "
						  << yoloNet.t.get_timeval_f(0) << " ms, infer: " << yoloNet.t.get_timeval_f(1) << " ms, post: " << yoloNet.t.get_timeval_f(2) << " ms"
						  << std::endl;
			}
			_img = detect::draw_boxes(_img, objs, 20, true);
			cv::imwrite("result.jpg", _img);
			if (debug) std::cout << "image result saved to result.jpg" << std::endl;
		}
		else {
			std::cerr << "image is empty!" << std::endl;
		}
		return 0;
	}

	cv::VideoCapture cap;
	cv::VideoWriter writer;

	if (pyStr::isdigit(source)) cap.open(std::stoi(source));
	else cap.open(source);

	int fps = 0;
	int num_frame = 0;
	cv::Size frameSize;
	bool save_video = false;
	if (cap.isOpened()) {
		fps = cap.get(cv::CAP_PROP_FPS);
		num_frame = cap.get(cv::CAP_PROP_FRAME_COUNT);
		if (!fps) fps=30;
		frameSize = cv::Size(
			static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
			static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
		);
		if (output.length()) {
			if (!pyStr::endswith(output, ".mp4") && !pyStr::endswith(output, ".MP4")) output += ".mp4";
			save_video = true;
			writer = cv::VideoWriter(output, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frameSize, true);
			if (!writer.isOpened()) {
				std::cerr << "Error opening VideoWriter" << std::endl;
				return -1;
			}
		}
	}


	bool flag=true;
	int frame_id = 0;
	while(cap.isOpened()) {

		flag = cap.read(_img);
		flag = flag && !_img.empty();
		if (!flag) break;

		tc.tic(0);
		objs = yoloNet.infer(_img);
		tc.toc(0);
		_img = detect::draw_boxes(_img, objs, 20, true);
		std::cout << "frame: " << pyStr::zfill(++frame_id, std::to_string(num_frame).length()) << "/" << num_frame
				  << ", num objects: " << objs.size() << ", time total:" << tc.get_timeval_f(0) << " ms, pre: "
				  << yoloNet.t.get_timeval_f(0) << " ms, infer: " << yoloNet.t.get_timeval_f(1) << " ms, post: " << yoloNet.t.get_timeval_f(2) << " ms"
				  << std::endl;

		if (save_video) writer.write(_img);

	}

	if (writer.isOpened()) writer.release();
	cap.release();

	return 0;
}
