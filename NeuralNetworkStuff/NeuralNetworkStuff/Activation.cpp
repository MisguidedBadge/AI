// Activation functions to be used in layers

#include "Activation.h"

// sigmoid function definition
float Sigmoid(float x)
{
	float val;
	val = float(1 / (1 + abs(x)));
	return val;
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
	float val;
	val = Sigmoid(x) * (1 - Sigmoid(x));
	return val;
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