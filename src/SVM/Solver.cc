#include "Solver.h"
#include <algorithm> /* min, max */
#include <armadillo>
#include <assert.h>
#include <stdlib.h> /* abs */

using namespace arma;

double SmoSolver::SvmOutputOnPoint(int i) {
  vec point = this->x.row(i).t();
  double result = this->kernel->KernelFunction(this->theta, point);
  return result - this->b;
}

double SmoSolver::KernelCal(int i1; int i2; bool onlyKernel){
  vec point1 = this->x.row(i1).t();
  vec point2 = this->x.row(i2).t();
  double result = this->kernel->KernelFunction(point1, point2);
  if(onlyKernel){
    return result
  }
  return result - this->b;
}

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

  e1 = this->SvmOutputOnPoint(i1) - y1;
  e2 = this->SvmOutputOnPoint(i1) - y2;
  s = y1 * y2;
  if (y1 != y2) {
    double temp = alpha2 - alpha1;
    low = std::max(0, temp);
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
  k11 = this->KernelCal(i1, i1,true);
  k12 = this->KernelCal(i1, i2,true);
  k22 = this->KernelCal(i2, i2,true);
  eta = k11 + k22 - 2 * k12;

  if (eta > 0) {
    a2 = alpha2 + y2 * (e1 - e2) / eta;
    if (a2 < low) {
      a2 = low;
    } else if (a2 > high) {
      a2 = high;
    }
  } else {
    // In papaer 2.1 (19)
    double f1 = y1 * (e1 + this->b) - alpha1 * k11 - s * alpha2 * k12;
    double f2 = y2 * (e2 + this->b) - s * alpha1 * k12 - alpha2 * k12;
    double low1 = alpha1 + s * (alpha2 - low);
    double high1 = alpha1 + s * (alpha2 - high);
    double objLow = low1 * f1 + low * f2 + 0.5 * low1 ^ 2 * k11 + 0.5 * low ^
                    2 * k22 + s * low * low1 * k12;
    double objHigh = high1 * f1 + high * f2 + 0.5 * high1 ^
                     2 * k11 + 0.5 * high ^
                     2 * k22 + s * high * high1 * K(i1, i2);
    if (objLow < objHigh - this->eps) {
      a2 = low;
    } else if (objLow > objHigh + this->eps) {
      a2 = high;
    } else {
      a2 = alpha2;
    }

    if (std::abs(a2 - alpha2) < this->eps * (a2 + alpha2 + this->eps)) {
      return 0;
    }
    a1 = alpha1 + s * (alpha2 - a2);

    // Update threshold to reflect change in Lagrange multipliers
    double b1 = 0.0;
    double b2 = 0.0;
    double bReal = 0.0;
    double temp1 = y1 * (a1 - alpha1);
    double temp2 = y2 * (a2 - alpha2);
    if (a1 > 0 && a1 < this->C) {
      bReal = e1 + temp1 * k11 + temp2 * k12 + b;
    } else if (a2 > 0 && a2 < this->C) {
      bReal = e2 + temp1 * k12 + temp2 * k22 + b;
    } else {
      b1 = e1 + temp1 * k11 + temp2 * k12 + b;
      b2 = e2 + temp1 * k12 + temp2 * k22 + b;
      bReal = (b1 + b2) / 2.0;
    }
    this->b = bReal;

    // Update weight vector (theta) to reflect change in al & a2, if SVM is
    // linear
    if (this->kernel->KernelType == LINEAR) {
      this->theta = this->theta + temp1 * this->x.row(i1).t() +
                    temp2 * this->x.row(i2).t();
    }
    // Update error cache using new Lagrange multipliers
    // TODO
    _error_cache[i1] = 0.0;
    _error_cache[i2] = 0.0;
    // Store a1, a2 in the alpha array
    this->lagrangeMultiplier[i1] = a1;
    this->lagrangeMultiplier[i2] = a2;
    return 1;
  }
}
