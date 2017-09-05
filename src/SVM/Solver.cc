#include "Solver.cc"
#include <armadillo>

int SvmSolver::TakeStep(int i1, int i2) {
  double alpha1 = 0.0, alpha2 = 0.0;
  int y1 = 0, y2 = 0;
  double e1 = 0.0, e2 = 0.0;

  if (i1 == i2) {
    return 0;
  }

  alpha1 = this->lagrangeMultiplier[i1];
  y1 = this->y[i1];
  // e1 =
}
