#include "Layer.h"

#include <Math.h>

// 2 rows (equal number of neurons)
// two inputs per row

// 2x2 matrix - layer
// overloaded constructor
Layer::Layer (int num_neurons,
              vector<float> inputs, 
              float Activation(float x),
              float learning_rate) {
  // define the number of inputs 
  this->inputs = inputs;
  this->num_inputs = inputs.size;
  this->num_neurons = num_neurons;
  this->learning_rate = learning_rate;
  
  // set up our activation function for the layer
  this->Activate = Activation;
}


Layer::~Layer() {
  this->weights.erase;
  this->DZW.erase;
  this->DLZ.erase;
  this->inputs.erase;
  this->Z.erase;
  this->outputs.erase;
}

// initialize weights to random values
void Layer::InitializeWeights(int inputs, int neurons) {
  for (int i = 0; i < neurons; i++) {
    for (int j = 0; j < inputs; j++) {
      this->weights[i][j] = rand()%6 + 1;
    }
  }
}

// instantiate layer workflow from input to output
void Layer::FeedForward () {
  this->ComputeZ();
  this->Activate();
}

void Layer::ComputeZ() {
  // zero Z vector
  for (int i = 0; i < this->num_neurons; i++)
    this->Z[i] = 0.0;

  // cycling through the neurons and the inputs
  // computing Z
  for (int i = 0; i < this->weights.size; i++) {
    for (int j = 0; j < this->weights[i].size; j++) {
      this->Z[i] += this->weights[i][j] * this->inputs[j];
    }
  }
}

void Layer::ActivateZ() {
  // run our z array into the activation function 
  for (int i = 0; i < this->num_neurons; i++) {
    this->outputs[i] = this->Activate(this->Z[i]);
  }
}

void Layer::BackPropagation(float target) {
  OutputError(target);
  ActivationDerivative();

  // determining the weight error
  for (int i = 0; i < this->weights.size; i++)  {
    for (int j = 0; j < this->weights[i].size; j++) {
      this->DZW[i][j] = this->DCL * this->DLZ[i] * inputs[i];
      // determining what value we need to change the weight
      this->weights[i][j] -= this->learning_rate * this->DZW[i][j];
    }
  }
}

// OutputError = Output - Target
void Layer::OutputError(float target) {
  int error = 0;

  for (int i = 0; i < this->num_outputs; i++)
    error += this->outputs[i] - target;

  this->DCL = error / this->num_outputs;
}

// ActivationDerivative = Derivative Activation Function passing Z
void Layer::ActivationDerivative () {
  
  for (int i = 0; i < num_neurons; i++) 
    this->DLZ[i] = DRelu(this->Z[i]);
}