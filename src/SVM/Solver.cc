#include "Solver.cc"
#include <armadillo>

int SvmSolver::TakeStep(int i1, int i2) {
  double alpha1 = 0.0;
  double alpha2 = 0.0;

  if (i1 == i2) {
    return 0;
  }

  alpha1 = LagrangeMultiplier[i1];
}
