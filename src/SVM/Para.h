#ifndef MODEL_SUPPORTVECTORMACHINE_PARAMETER_H_
#define MODEL_SUPPORTVECTORMACHINE_PARAMETER_H_
#include "Kernel.h"

struct SvmParameter {
  KernelType kernelType;

  // For POLY
  // int degree;
  // For POLY and SIGMOID
  // double coef0;

  // For Poly RBF AND SIGMOID
  double gamma;

  // For Poly and SIGMOID
  // double coef0;
};

#endif
