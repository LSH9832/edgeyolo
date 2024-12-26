// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_PLUGIN_HB_DNN_PLUGIN_H_
#define DNN_PLUGIN_HB_DNN_PLUGIN_H_

#include <string>

#include "./hb_dnn_layer.h"

typedef hobot::dnn::Layer *(*hbDNNLayerCreator)();

/**
 * Register layer creator function
 * @param[in] layerName: layer type
 * @param[in] layerCreator: layer creator function
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNRegisterLayerCreator(char const *layerType,
                                  hbDNNLayerCreator layerCreator);

/**
 * Unregister layer creator function
 * @param[in] layerName: layer type
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNUnregisterLayerCreator(char const *layerType);

#endif  // DNN_PLUGIN_HB_DNN_PLUGIN_H_
