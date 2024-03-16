/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2012-2021. All rights reserved.
 * Description: header for davinci net
*/

#ifndef DAVINCI_NET_H
#define DAVINCI_NET_H

#include <detect/mdc/blob.h>

#include <iostream>
#include <map>
#include <memory>

class DavinciNet {
private:
    uint32_t modelId_;
    void *modelMemPtr_;
    void *modelWeightPtr_;
    aclmdlDesc *modelDesc_;
    aclmdlDataset *input_;
    aclmdlDataset *output_;

    std::map<std::string, std::shared_ptr<Blob<void>>> blobMap;

    int32_t deviceId_;
    aclrtContext context_;
    aclrtStream stream_;

    bool isLoaded_;

public:

    std::vector<std::string> input_names, output_names;
    std::vector<std::vector<int>> inputs_dims, outputs_dims;

    DavinciNet();

    ~DavinciNet();

    int Init(const std::string &modelPath, const int32_t deviceId = 0);

    int CreateContextAndStream();

    int LoadModelFromFile(const std::string &modelPath);
    int CreateDesc();
    int CreateInputTensor();
    int CreateOutputTensor();

    void DestroyDesc();
    void DestroyInputTensor();
    void DestroyOutputTensor();
    void UnloadModel();
    void DestroyContextAndStream();
    void DisplayTensorInfo(const aclmdlIODims &dims, const aclDataType dtype, const bool input);

    int Inference();
    std::shared_ptr<BlobDesc> GetBlob(const std::string &blobName);
};

#endif
