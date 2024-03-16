/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2021. All rights reserved.
 * Description: header for dvpp handler
*/

#ifndef DVPP_HANDLER_H
#define DVPP_HANDLER_H

#include <detect/mdc/image_data.h>

#include <string>
#include <vector>
#include <map>

#include <dvpp/dvpp_config.h>
#include <dvpp/idvppapi.h>
#include <dvpp/Vpc.h>

class DVPPHandler {
private:
    int ReadImage(const std::string &fileName, Blob<uint8_t> &inBuf);

    int DvppProcess(const int32_t cmd);

    int ConfigStride(const ImageData &imgData);

    inline int CheckOdd(const int num)
    {
        return num % 2 != 0 ? num : num - 1;
    }

    inline int CheckEven(const int num)
    {
        return num % 2 == 0 ? num : num - 1;
    }

    VpcUserImageConfigure image_config;
    VpcUserRoiConfigure roi_config;
    VpcUserRoiInputConfigure *input_config;
    VpcUserRoiOutputConfigure *output_config;

    dvppapi_ctl_msg dvppApiCtlMsg;
    IDVPPAPI *pidvppapi = nullptr;

public:
    DVPPHandler();

    ~DVPPHandler();

    int DecodePNG(const std::string &infile_name, ImageData &imgdata);

    int DecodeJPEG(const std::string &infile_name, ImageData &imgdata);

    // Roi: {x1, y1, x2, y2}
    int Resize(ImageData &srcImg, const std::vector<int> &cropArea,
               ImageData &dstImg, const std::vector<int> &outputArea);
};

#endif
