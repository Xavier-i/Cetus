#ifndef MODEL_SUPPORTVECTORMACHINE_SOLVER_H_
#define MODEL_SUPPORTVECTORMACHINE_SOLVER_H_
#include "Kernel.h"
#include "Para.h"
#include <armadillo>

class SmoSolver {
public:
  SvmSolver();
  ~SvmSolver();
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

  arma::vec lagrangeMultiplier;
  arma::vec errorCache;
  arma::vec y;
};

#endif
