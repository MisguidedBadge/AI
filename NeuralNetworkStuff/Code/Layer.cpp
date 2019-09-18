#include "Layer.h"

// 2 rows (equal number of neurons)
// two inputs per row

// 2x2 matrix - layer
// overloaded constructor
Layer::Layer(int neurons, int inputs, void* Activation(float x)) {
  // set up our activation function for the layer
  this->Activate = Activation;
}


Layer::~Layer() {

}

// initialize weights to random values
void Layer::Initialize_Weights(int inputs, int neurons) {
  for (int i = 0; i < neurons; i++) {
    for (int j = 0; j < inputs; j++) {
      this->weights[i][j] = rand()%6 + 1;
    }
  }
}

// instantiate layer workflow from input to output
Layer::FeedForward () {
  
}


