#include "Kernel.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

Kernel::Kernel(int l, svm_node *const *x_, const svm_parameter &param)
    : kernelType(param.kernel_type), degree(param.degree), gamma(param.gamma),
      coef0(param.coef0) {
  switch (kernel_type) {
    if (this->kernelType == LINEAR) {
      kernel_function = &Kernel::KernelLinear;
    }

    if (this->kernelType == POLY) {
      kernel_function = &Kernel::KernelPoly;
    } else if (this->kernelType == RBF) {
      kernel_function = &Kernel::KernelRBF;
    } else if (this->kernelType == SIGMOID) {
      kernel_function = &Kernel::kernel_sigmoid;
    } else if (this->kernelType == KernelSigmoid) {
      kernel_function = &Kernel::kernel_precomputed;
    }
  }

  /*clone(x, x_, l);

  if (kernel_type == RBF) {
    x_square = new double[l];
    for (int i = 0; i < l; i++)
      x_square[i] = dot(x[i], x[i]);
  } else
    x_square = 0;*/
}
