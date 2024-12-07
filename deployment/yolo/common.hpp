/*
 * common.h
 *
 *  Created on: Mar 6, 2024
 *      Author: LSH9832
 */
#ifndef YOLO_COMMON_HPP
#define YOLO_COMMON_HPP

#include <opencv2/opencv.hpp>

namespace detect {

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

    struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
    };

    struct resizeInfo
    {
        cv::Mat resized_img;
        float factor;
    };

    template <typename T>
    T sigmoid(T x) {
		return 1 / (1 + expf(-x));
	}

    template <typename T>
	T unsigmoid(T x) {
		return -logf((1 - x) / x);
	}

    std::vector<std::string> names;

    cv::Scalar get_color(int index){
        index %= 20;
        return cv::Scalar(color_list[index][2], color_list[index][1], color_list[index][0]);
    }

    void static_resize(cv::Mat& img, cv::Size input_size, resizeInfo& _info) {
        float r = std::min(input_size.width / (img.cols*1.0), input_size.height / (img.rows*1.0));
        int unpad_w = (int)round(r * img.cols);
        int unpad_h = (int)round(r * img.rows);

        cv::Mat re(unpad_h, unpad_w, CV_8UC3);
        cv::resize(img, re, re.size());
        _info.resized_img = cv::Mat(input_size.height, input_size.width, CV_8UC3, cv::Scalar(114, 114, 114));
        re.copyTo(_info.resized_img(cv::Rect(0, 0, re.cols, re.rows)));
        _info.factor = 1./ r;
    }

    resizeInfo resizeAndPad(cv::Mat &src_img, cv::Size _size, bool center=false, bool show=false)
    {
        int img_C = src_img.channels();
        int img_H = src_img.rows;
        int img_W = src_img.cols;
        cv::Mat resize_img;

        resizeInfo info;

        float factor_h = (double)img_H / _size.height;
        float factor_w = (double)img_W / _size.width;

        // std::cout << factor_h << "," << factor_w << std::endl;

        if (factor_h >= factor_w)
        {
            info.factor = factor_h;
            int new_W = int(img_W / factor_h);
            if (new_W % 2 != 0) new_W -= 1;
            cv::resize(src_img.clone(), resize_img, cv::Size(new_W, _size.height));
            int img_C2 = resize_img.channels();
            int img_H2 = resize_img.rows;
            int img_W2 = resize_img.cols;

            int pad_w0 = 0;
            int pad_w1 = _size.width - new_W;
            if (center) {
                pad_w1 /= 2;
                pad_w0 = pad_w1;
            }

            cv::copyMakeBorder(resize_img, info.resized_img, 0, 0, pad_w0, pad_w1, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }
        else
        {
            info.factor = factor_w;
            int new_H = int(img_H / factor_w);
            if (new_H % 2 != 0) new_H -= 1;

            cv::resize(src_img.clone(), resize_img, cv::Size(_size.width, new_H));
            int img_C2 = resize_img.channels();
            int img_H2 = resize_img.rows;
            int img_W2 = resize_img.cols;

            int pad_h0 = 0;
            int pad_h1 = _size.height - new_H;
            if (center) {
                pad_h1 /= 2;
                pad_h0 = pad_h1;
            }

            cv::copyMakeBorder(resize_img, info.resized_img, pad_h0, pad_h1, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }
        if (show) {
            cv::imshow("resize pad image", info.resized_img);
            cv::waitKey(0);
        }
        return info;
    }

    cv::Mat draw_boxes(cv::Mat image,
                       std::vector<Object> &objects,
                       int draw_size=20,
                       bool draw_label=true) {
        cv::Mat d_img = image.clone();
        cv::Scalar color;
        cv::Scalar txt_color;
        cv::Scalar txt_bk_color;
        cv::Size label_size;
        int baseLine = 0;
        int x, y, out_point_y, w, h;
        int line_thickness = std::round((double)draw_size / 10.0);

        for(int k=0; k<objects.size(); k++){
            color = get_color(objects.at(k).label);

            x = objects.at(k).rect.x;
            y = objects.at(k).rect.y;
            w = objects.at(k).rect.width;
            h = objects.at(k).rect.height;

            cv::rectangle(d_img,
                          objects.at(k).rect,
                          color,
                          line_thickness);

            if (draw_label){
                txt_color = (cv::mean(color)[0] > 127)?cv::Scalar(0, 0, 0):cv::Scalar(255, 255, 255);
                std::string label = names.at(objects.at(k).label) + " " + std::to_string(objects.at(k).prob).substr(0, 4);
                label_size = cv::getTextSize(label.c_str(), cv::LINE_AA, double(draw_size) / 30.0, (line_thickness>1)?line_thickness-1:1, &baseLine);
                txt_bk_color = color; // * 0.7;
                y = (y > d_img.rows)?d_img.rows:y + 1;
                out_point_y = y - label_size.height - baseLine;
                if (out_point_y >= 0) y = out_point_y;
                cv::rectangle(d_img, cv::Rect(cv::Point(x - (line_thickness - 1), y), cv::Size(label_size.width, label_size.height + baseLine)),
                            txt_bk_color, -1);
                cv::putText(d_img, label, cv::Point(x, y + label_size.height),
                            cv::LINE_AA, double(draw_size) / 30.0, txt_color, (line_thickness>1)?line_thickness-1:1);
            }

        }
        return d_img;
    }

    static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) {
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

    static void qsort_descent_inplace(std::vector<Object>& objects) {
        if (objects.empty())
            return;

        qsort_descent_inplace(objects, 0, objects.size() - 1);
    }

    static inline float intersection_area(const Object& a, const Object& b) {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++) areas[i] = faceobjects[i].rect.area();

        for (int i = 0; i < n; i++) {
            const Object& a = faceobjects[i];

            bool keep = true;
            for (int j = 0; j < (int)picked.size(); j++) {
                const Object& b = faceobjects[picked[j]];
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold) {
                    keep = false;
                    break;
                }
            }

            if (keep) picked.push_back(i);
        }
    }

}


#endif
#define YOLO_COMMON_HPP
