#ifndef MODEL_SUPPORTVECTORMACHINE_KERNEL_H_
#define MODEL_SUPPORTVECTORMACHINE_KERNEL_H_
#include <armadillo>

enum KernelType { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

class Kernel {
public:
  Kernel();
  ~Kernel();

  static double k_function();

  // svm_parameter
  const int kernel_type;
  const int degree;
  const double gamma;
  const double coef0;

  double kernel_linear(int i, int j) const { return dot(x[i], x[j]); }
  double kernel_poly(int i, int j) const {
    return powi(gamma * dot(x[i], x[j]) + coef0, degree);
  }
  double kernel_rbf(int i, int j) const {
    return exp(-gamma * (x_square[i] + x_square[j] - 2 * dot(x[i], x[j])));
  }
  double kernel_sigmoid(int i, int j) const {
    return tanh(gamma * dot(x[i], x[j]) + coef0);
  }
  double kernel_precomputed(int i, int j) const {
    return x[i][(int)(x[j][0].value)].value;
  }
};

#endif
