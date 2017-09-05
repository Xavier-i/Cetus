#include "Solver.cc"
#include <algorithm> /* min, max */
#include <armadillo>
#include <stdlib.h> /* abs */

int SmoSolver::TakeStep(int i1, int i2) {
  double alpha1 = 0.0, alpha2 = 0.0;
  double a1 = 0.0, a2 = 0.0;
  int y1 = 0, y2 = 0;
  double e1 = 0.0, e2 = 0.0;
  int s = 0;
  double low = 0.0, high = 0.0;
  double k11 = 0.0, k12 = 0.0, k22 = 0.0, eta = 0.0;

  if (i1 == i2) {
    return 0;
  }

  alpha1 = this->lagrangeMultiplier[i1];
  alpha2 = this->lagrangeMultiplier[i2];
  y1 = this->y[i1];
  y2 = this->y[i2];

  // Havent developed
  e1 = SvmOutputOnPoint(i1) - y1;
  e2 = SvmOutputOnPoint(i1) - y2;
  s = y1 * y2;
  if (y1 != y2) {
    double temp = alpha2 - alpha1 low = std::max(0, temp);
    high = std::min(this->C, this->C + temp);
  } else {
    double temp = alpha2 + alpha1;
    low = std::max(0, temp - this->C);
    high = std::min(this->C, temp);
  }

  // check if low is equal to high
  if (abs(low - high) < 1.0e-7) {
    return 0;
  }
  k11 = this->kernel->KernelFunction(i1, i1);
  k12 = this->kernel->KernelFunction(i1, i2);
  k22 = this->kernel->KernelFunction(i2, i2);
  eta = 2 * k12 - k11 - k22;
  if (eta > 0) {
    a2 = alpha2 + y2 * (e1 - e2) / eta;
    if (a2 < low) {
      a2 = low;
    } else if (a2 > high) {
      a2 = high;
    }
  } else {
  }
}
