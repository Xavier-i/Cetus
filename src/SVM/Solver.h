#ifndef MODEL_SUPPORTVECTORMACHINE_SOLVER_H_
#define MODEL_SUPPORTVECTORMACHINE_SOLVER_H_
#include "Kernel.h"
#include "Para.h"
#include <armadillo>

class SvmSolver {
public:
  SvmSolver();
  ~SvmSolver();

private:
  int TakeStep(int i1, int i2);
  int ExamineExample(int i2);

  arma::vec LagrangeMultiplier;
};

#endif
