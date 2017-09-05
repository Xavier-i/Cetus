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

private:
  int TakeStep(int i1, int i2);
  int ExamineExample(int i2);

  double C;
  arma::vec lagrangeMultiplier;
  arma::vec errorCache;
  arma::vec y;
};

#endif
