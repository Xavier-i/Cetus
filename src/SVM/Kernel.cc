#include "Kernel.h"
#include <armadillo>
#include <assert.h>
#include <iostream>
#include <math.h>

using namespace arma;

Kernel::Kernel(KernelType type) : kernelType(type) {

  if (this->kernelType == LINEAR) {
    KernelFunction = &Kernel::KernelLinear;
  } else if (this->kernelType == RBF) {
    KernelFunction = &Kernel::KernelRBF;
  } /* else if (this->kernelType == POLY) {
     KernelFunction = &Kernel::KernelPoly;
   } else if (this->kernelType == SIGMOID) {
     KernelFunction = &Kernel::kernel_sigmoid;
   } else if (this->kernelType == KernelSigmoid) {
     KernelFunction = &Kernel::kernel_precomputed;
   }*/
}

/*
double Kernel::KernelLinear(int i1, int i2) {
  return this->KernelLinear(this->x.rows(i1).t(),this->x.rows(i2).t()); }
double Kernel::KernelRBF(vec x, vec y) {
  return this->KernelRBF(this->x.rows(i1).t(),this->x.rows(i2).t());
}
*/
double Kernel::KernelLinear(vec &x1, vec &x2) const {
  return arma::dot(x1, x2);
}
double Kernel::KernelRBF(vec &x1, vec &x2) const {
  double temp =norm(x1 - x2);
  return exp(-this->gamma * pow(temp,2.0));
}
