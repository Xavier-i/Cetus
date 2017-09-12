#ifndef MODEL_SUPPORTVECTORMACHINE_PARAMETER_H_
#define MODEL_SUPPORTVECTORMACHINE_PARAMETER_H_

enum KernelType { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct SvmParameter {

  SvmParameter(KernelType type = LINEAR, double regPar = 1.0,
               double gamma = 1.0)
      : kernelType{type}, regParameterC{regPar}, gamma{gamma} {}
  ~SvmParameter() {}
  KernelType kernelType;

  double regParameterC;

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
