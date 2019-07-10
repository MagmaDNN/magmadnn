/**
 * @file layers.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-27
 * 
 * @copyright Copyright (c) 2019
 */
#pragma once

/* Load in all the layer headers */

#include "input/inputlayer.h"
#include "fullyconnected/fullyconnectedlayer.h"
#include "output/outputlayer.h"
#include "activation/activationlayer.h"
#include "dropout/dropoutlayer.h"
#include "conv2d/conv2dlayer.h"
#include "pooling/poolinglayer.h"
#include "flatten/flattenlayer.h"

#include "layer/wrapper.h"
