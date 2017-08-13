#include "Kernel.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

// linear: u'*v
// polynomial: (gamma*u'*v + coef0)^degree
// radial basis function: exp(-gamma*|u-v|^2)
// sigmoid: tanh(gamma*u'*v + coef0)
// precomputed kernel (kernel values in training_set_file)
SupportVectorMahchine::KernelFuc(mat &xi, mat &xj) {
  if (this->kernel == LINEAR) {
  } else if (this->kernel == POLY) {
  } else if (this->kernel == RBF) {
  } else if (this->kernel == SIGMOID) {
  } else if (this->kernel == PRECOMPUTED) {
  } else {
    std::cerr << "Not a valid kernel" << std::endl;
    exit 404;
  }
}
