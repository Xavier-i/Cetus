#include "Kernel.h"
#include "Para.h"
#include <armadillo>
#include <assert.h>
#include <iostream>
#include <math.h>

using namespace arma;

Kernel::Kernel(int l, svm_node *const *x_, const SvmParameter &param)
    : kernelType(param.kernelType), degree(param.degree), gamma(param.gamma),
      coef0(param.coef0) {
  switch (kernel_type) {
    if (this->kernelType == LINEAR) {
      kernel_function = &Kernel::KernelLinear;
    } else if (this->kernelType == RBF) {
      kernel_function = &Kernel::KernelRBF;
    } /* else if (this->kernelType == POLY) {
       kernel_function = &Kernel::KernelPoly;
     } else if (this->kernelType == SIGMOID) {
       kernel_function = &Kernel::kernel_sigmoid;
     } else if (this->kernelType == KernelSigmoid) {
       kernel_function = &Kernel::kernel_precomputed;
     }*/
  }

  /*clone(x, x_, l);

  if (kernel_type == RBF) {
    x_square = new double[l];
    for (int i = 0; i < l; i++)
      x_square[i] = dot(x[i], x[i]);
  } else
    x_square = 0;*/
}

vec Kernel::KernelLinear(mat X, vec y) { return x * y; }
vec Kernel::KernelRBF(mat X, vec y) {
  int rowNum = X.n_rows;
  vec result = zeros<vec>(rowNum);
  for (unsigned int i = 0; i < rowNum; i++) {
    result[i] = exp(-this->gamma * norm((x.row(i).t() - y), 2) ^ 2);
  }
  return result
}
