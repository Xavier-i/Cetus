#ifndef CETUS_SUPPORTVECTORMACHINE_KERNEL_H_
#define CETUS_SUPPORTVECTORMACHINE_KERNEL_H_

#include "Para.h"
#include <armadillo>

using namespace arma;
class Kernel {
public:
  Kernel(SvmParameter *para);
  ~Kernel(){};

  // Function Pointer
  // double (*KernelFunction)(int i, int j);
  double (Kernel::*KernelFunction)(vec x1, vec x2) const;
  // svm_parameter
  const KernelType kernelType;
  double gamma;

  // linear: u'*v
  // polynomial: (gamma*u'*v + coef0)^degree
  // radial basis function: exp(-gamma*|u-v|^2)
  // sigmoid: tanh(gamma*u'*v + coef0)
  // precomputed kernel (kernel values in training_set_file)

  double KernelLinear(vec x1, vec x2) const;
  double KernelRBF(vec x1, vec x2) const;
  /*
    vec KernelLinear(int i1, int i2) const;
    vec KernelRBF(int i1, int i2) const;
  */
};

#endif
