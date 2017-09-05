#ifndef MODEL_SUPPORTVECTORMACHINE_KERNEL_H_
#define MODEL_SUPPORTVECTORMACHINE_KERNEL_H_
#include "Para.h"
#include <armadillo>

enum KernelType { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */
using namespace arma;
class Kernel {
public:
  Kernel(const SvmParameter &param);
  ~Kernel();

  // static double k_function();

  // Function Pointer
  double (Kernel::*KernelFunction)(int i, int j) const;
  // svm_parameter
  const KernelType kernelType;
  const int degree;
  const double gamma;
  const double coef0;

  // linear: u'*v
  // polynomial: (gamma*u'*v + coef0)^degree
  // radial basis function: exp(-gamma*|u-v|^2)
  // sigmoid: tanh(gamma*u'*v + coef0)
  // precomputed kernel (kernel values in training_set_file)

  vec KernelLinear(mat x, vec y) const;
  vec KernelRBF(mat x, vec y) const;

  vec KernelLinear(vec x1, vec x2) const;
  vec KernelRBF(vec x1, vec x2) const;
};

#endif
