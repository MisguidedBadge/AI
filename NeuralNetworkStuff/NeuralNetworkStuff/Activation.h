#ifndef ACT_H
#define ACT_H

#include "libraries.h"

// activation functions
float Sigmoid(float x);
float Relu(float x);

// activation function derivatives
float DSigmoid(float x);
float DRelu(float x);
#endif
