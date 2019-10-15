// Activation functions to be used in layers

#include "Activation.h"

// sigmoid function definition
float Sigmoid(float x)
{
	return 1 / (1 + exp(-x));
}

// relu function definition
float Relu(float x)
{
	if (x > 0)
	{
		return x;
	}
	else
	{
		return 0;
	}
}

// derivative of sigmoid
float DSigmoid(float x)
{
	return Sigmoid(x) * (1 - Sigmoid(x));
}

// derivative of Relu
float DRelu(float x)
{
	if (x > 0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}