/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2021. All rights reserved.
 * Description: header for class image data
*/

#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#include <detect/mdc/blob.h>
#include <memory>

enum PngColorSpace {
    DVPP_PNG_DECODE_OUT_RGB = 6,
    DVPP_PNG_DECODE_OUT_RGBA = 7
};

class ImageData {
public:
    ImageData(const uint32_t h = 0, const uint32_t w = 0, const uint32_t fmt = -1)
              : height(h), width(w), format(fmt)
    {
        rawData = std::make_shared<Blob<uint8_t>>();
    }

    ImageData(const uint32_t h, const uint32_t w, std::shared_ptr<Blob<uint8_t>> &data)
              : height(h), width(w), rawData(data) {}

public:
    uint32_t height;
    uint32_t width;
    int format;
    std::shared_ptr<Blob<uint8_t>> rawData;
};

#endif
