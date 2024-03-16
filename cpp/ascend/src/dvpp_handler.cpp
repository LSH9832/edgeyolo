/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2021. All rights reserved.
 * Description: source for dvpp handler
*/

#include <detect/mdc/dvpp_handler.h>

int DVPPHandler::DecodePNG(const std::string &fileName, ImageData &imgData)
{
    // read png file
    Blob<uint8_t> inBuf;
    if (ReadImage(fileName, inBuf) != 0) {
        ACL_APP_LOG(ACL_ERROR, "Read PNG image failed!");
        return 1;
    }

    // prepare msg
    struct PngInputInfoAPI inputPngData;
    struct PngOutputInfoAPI outputPngData;
    inputPngData.inputData = inBuf.Data();
    inputPngData.inputSize = inBuf.Size();
    inputPngData.transformFlag = 0;

    dvppApiCtlMsg.in       = (void *)(&inputPngData);
    dvppApiCtlMsg.in_size  = sizeof(struct PngInputInfoAPI);
    dvppApiCtlMsg.out      = (void *)(&outputPngData);
    dvppApiCtlMsg.out_size = sizeof(struct PngOutputInfoAPI);

    int ret = DvppProcess(DVPP_CTL_PNGD_PROC);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Decode PNG image failed!");
        outputPngData.FreeOutputMemory();
        return 1;
    }

    const int rgbFormat = 2;
    imgData.format = outputPngData.format == rgbFormat ? DVPP_PNG_DECODE_OUT_RGB : DVPP_PNG_DECODE_OUT_RGBA;
    imgData.height = outputPngData.high;
    imgData.width = outputPngData.width;
    ret = imgData.rawData->Alloc(outputPngData.outputSize);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Alloc image data failed!");
        return 1;
    }
    ret = memcpy_s(imgData.rawData->Data(), imgData.rawData->Size(),
                   outputPngData.outputData, outputPngData.outputSize);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Memcpy image data failed!");
        outputPngData.FreeOutputMemory();
        return 1;
    }

    outputPngData.FreeOutputMemory();
    return 0;
}

int DVPPHandler::DecodeJPEG(const std::string &fileName, ImageData &imgData)
{
    // read Jpeg file
    Blob<uint8_t> inBuf;
    if (ReadImage(fileName, inBuf) != 0) {
        ACL_APP_LOG(ACL_ERROR, "Read PNG image failed!");
        return 1;
    }

    const int extendLen = 8;
    struct JpegdIn jpegdInData;
    jpegdInData.jpegData = inBuf.Data();
    jpegdInData.jpegDataSize = inBuf.Size() + extendLen;
    jpegdInData.isYUV420Need = false;
    jpegdInData.isVBeforeU = false;

    struct JpegdOut jpegdOutData;
    int ret = DvppGetOutParameter((void *)(&jpegdInData), (void *)(&jpegdOutData), GET_JPEGD_OUT_PARAMETER);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Dvpp get out parameter failed!");
        return 1;
    }

    Blob<uint8_t> outBuf;
    if (outBuf.Alloc(jpegdOutData.yuvDataSize) != 0) {
        ACL_APP_LOG(ACL_ERROR, "Alloc address for output image failed!");
        return 1;
    }
    jpegdOutData.yuvData = outBuf.Data();

    dvppApiCtlMsg.in = (void *)&jpegdInData;
    dvppApiCtlMsg.in_size = sizeof(struct JpegdIn);
    dvppApiCtlMsg.out = (void *)&jpegdOutData;
    dvppApiCtlMsg.out_size = sizeof(struct JpegdOut);

    ret = DvppProcess(DVPP_CTL_JPEGD_PROC);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Decode JPEG image failed!");
        return 1;
    }

    // construct image data
    imgData = ImageData(jpegdOutData.imgHeight, jpegdOutData.imgWidth, jpegdOutData.outFormat);
    ret = imgData.rawData->Alloc(jpegdOutData.yuvDataSize);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Alloc image data failed!");
        return 1;
    }
    ret = memcpy_s(imgData.rawData->Data(), imgData.rawData->Size(), jpegdOutData.yuvData, jpegdOutData.yuvDataSize);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Memcpy image data failed!");
        return 1;
    }

    return 0;
}

int DVPPHandler::ConfigStride(const ImageData &imgData)
{
    int channel = 1;
    const int widthAlign = 128;
    const int heightAlign = 16;
    const int rgbChan = 3;
    const int argbChan = 3;
    switch (imgData.format) {
        case DVPP_PNG_DECODE_OUT_RGB:
            channel = rgbChan;
            image_config.inputFormat = INPUT_RGB;
            break;
        case DVPP_PNG_DECODE_OUT_RGBA:
            channel = argbChan;
            image_config.inputFormat = INPUT_RGBA;
            break;
        case DVPP_JPEG_DECODE_OUT_YUV444:
            image_config.inputFormat = INPUT_YUV444_SEMI_PLANNER_UV;
            break;
        case DVPP_JPEG_DECODE_OUT_YUV422_H2V1:
            image_config.inputFormat = INPUT_YUV422_SEMI_PLANNER_UV;
            break;
        case DVPP_JPEG_DECODE_OUT_YUV420:
            image_config.inputFormat = INPUT_YUV420_SEMI_PLANNER_UV;
            break;
        case DVPP_JPEG_DECODE_OUT_YUV400:
            image_config.inputFormat = INPUT_YUV400;
            break;
        default:
            ACL_APP_LOG(ACL_ERROR, "Unsupport image colorspace!");
            return 1;
    }
    image_config.widthStride = ALIGN_UP(imgData.width * channel, widthAlign);
    image_config.heightStride = ALIGN_UP(imgData.height, heightAlign);

    return 0;
}

int DVPPHandler::Resize(ImageData &srcImg, const std::vector<int> &cropArea,
                        ImageData &dstImg, const std::vector<int> &outputArea)
{
    // input image configure
    image_config.bareDataAddr = srcImg.rawData->Data();
    image_config.bareDataBufferSize = srcImg.rawData->Size();
    image_config.isCompressData = false;
    image_config.outputFormat = OUTPUT_YUV420SP_UV;
    image_config.yuvSumEnable = false;
    image_config.cmdListBufferAddr = nullptr;
    image_config.cmdListBufferSize = 0;

    int ret = ConfigStride(srcImg);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Config src image stride failed!");
        return 1;
    }

    // ROI configure
    const int left = 0;
    const int top = 1;
    const int right = 2;
    const int bottom = 3;
    const int heightAlign = 2;
    const int widthAlign = 16;
    input_config->cropArea.leftOffset = CheckEven(cropArea[left]);
    input_config->cropArea.upOffset = CheckEven(cropArea[top]);
    input_config->cropArea.rightOffset = CheckOdd(cropArea[right]);
    input_config->cropArea.downOffset = CheckOdd(cropArea[bottom]);

    // output image configure
    output_config->addr = dstImg.rawData->Data();
    output_config->bufferSize = dstImg.rawData->Size();
    output_config->heightStride = ALIGN_UP(dstImg.height, heightAlign);
    output_config->widthStride = ALIGN_UP(dstImg.width, widthAlign);
    output_config->outputArea.leftOffset = CheckEven(outputArea[left]);
    output_config->outputArea.upOffset = CheckEven(outputArea[top]);
    output_config->outputArea.rightOffset = CheckOdd(outputArea[right]);
    output_config->outputArea.downOffset = CheckOdd(outputArea[bottom]);

    dvppApiCtlMsg.in = static_cast<void*>(&image_config);
    dvppApiCtlMsg.in_size = sizeof(VpcUserImageConfigure);

    ret = DvppProcess(DVPP_CTL_VPC_PROC);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Handle VPC failed!");
        return 1;
    }

    return 0;
}

DVPPHandler::DVPPHandler()
{
    // Struct for VPC
    input_config = &roi_config.inputConfigure;
    output_config = &roi_config.outputConfigure;
    roi_config.next = nullptr;
    image_config.roiConfigure = &roi_config;

    // create dvpp api
    int ret = CreateDvppApi(pidvppapi);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "VPC create dvppApi failed!");
        pidvppapi = nullptr;
    }
}

DVPPHandler::~DVPPHandler()
{
    if (pidvppapi != nullptr) {
        DestroyDvppApi(pidvppapi);
    }
}

int DVPPHandler::DvppProcess(const int32_t cmd)
{
    if (pidvppapi == nullptr) {
        ACL_APP_LOG(ACL_ERROR, "Can not get dvpp api!");
        return 1;
    }

    int ret = DvppCtl(pidvppapi, cmd, &dvppApiCtlMsg);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Call vpc dvppctl process faild!");
        return 1;
    }

    return 0;
}

int DVPPHandler::ReadImage(const std::string &fileName, Blob<uint8_t> &inBuf)
{
    FILE *fpIn = fopen(fileName.c_str(), "rb");
    if (fpIn == nullptr) {
        ACL_APP_LOG(ACL_ERROR, "Can not open file %s.", fileName.c_str());
        return 1;
    }
    fseek(fpIn, 0, SEEK_END);
    uint32_t fileLen = ftell(fpIn);
    fseek(fpIn, 0, SEEK_SET);

    int ret = inBuf.Alloc(fileLen);
    if (ret != 0) {
        ACL_APP_LOG(ACL_ERROR, "Alloc inbuf failed!");
        fclose(fpIn);
        return 1;
    }

    uint32_t len = fread(inBuf.Data(), 1, fileLen, fpIn);
    if (len != fileLen) {
        ACL_APP_LOG(ACL_ERROR, "Read file error!");
        fclose(fpIn);
        return 1;
    }

    fclose(fpIn);
    return 0;
}
