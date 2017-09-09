#ifndef MODEL_SUPPORTVECTORMACHINE_SOLVER_H_
#define MODEL_SUPPORTVECTORMACHINE_SOLVER_H_
#include "Kernel.h"
#include "Para.h"
#include <armadillo>
double tol = 1.0e-3;
class SmoSolver {
public:
  SmoSolver();
  ~SmoSolver();
  Kernel *kernel;
  arma::vec &theta;
  // First feature
  arma::mat &x;

  // Target feature
  // Elements in y have to be either 1 or 0
  arma::vec &y;

private:
  int TakeStep(int i1, int i2);
  int ExamineExample(int i2);

  double C;
  double b;
  double eps = 1.0e-7;
  double SvmOutputOnPoint(int i);
  double KernelCal(int i1, int i2, bool onlyKernel);
  arma::vec lagrangeMultiplier;
  arma::vec errorCache;
};

#endif
