#include "Solver.h"
#include <algorithm> /* min, max */
#include <armadillo>
#include <assert.h>
#include <iostream>
#include <functional>
#include <math.h>   /* pow */
#include <stdlib.h> /* abs, drand48 */

using namespace arma;

double SmoSolver::SvmOutputOnPoint(int i) {
  vec point = this->x.row(i).t();
  double result = (kernel->*(kernel->KernelFunction))(this->theta, point);
  return result - this->b;
}

double SmoSolver::Predict(vec x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  return dot(this->theta, x) - this->b;
}

double SmoSolver::KernelCal(int i1, int i2) {
  vec point1 = this->x.row(i1).t();
  vec point2 = this->x.row(i2).t();
  return (kernel->*(kernel->KernelFunction))(point1, point2);
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

  if (alpha1 > 0 && alpha1 < this->C) {
    e1 = this->errorCache[i1];
  } else {
    e1 = this->SvmOutputOnPoint(i1) - y1;
  }
  if (alpha2 > 0 && alpha2 < this->C) {
    e2 = this->errorCache[i2];
  } else {
    e2 = this->SvmOutputOnPoint(i1) - y2;
  }

  s = y1 * y2;
  if (y1 != y2) {
    double temp = alpha2 - alpha1;
    low = std::max(0.0, temp);
    high = std::min(this->C, this->C + temp);
  } else {
    double temp = alpha2 + alpha1;
    low = std::max(0.0, temp - this->C);
    high = std::min(this->C, temp);
  }

  // check if low is equal to high
  if (abs(low - high) < 1.0e-7) {
    return 0;
  }
  k11 = this->KernelCal(i1, i1);
  k12 = this->KernelCal(i1, i2);
  k22 = this->KernelCal(i2, i2);
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
    double objLow = low1 * f1 + low * f2 + 0.5 * pow(low1, 2.0) * k11 +
                    0.5 * pow(low, 2.0) * k22 + s * low * low1 * k12;
    double objHigh = high1 * f1 + high * f2 + 0.5 * pow(high1, 2.0) * k11 +
                     0.5 * pow(high, 2.0) * k22 + s * high * high1 * k12;
    if (objLow < objHigh - this->eps) {
      a2 = low;
    } else if (objLow > objHigh + this->eps) {
      a2 = high;
    } else {
      a2 = alpha2;
    }
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
  double bDiff = bReal - this->b;
  this->b = bReal;

  // Update weight vector (theta) to reflect change in al & a2, if SVM is
  // linear
  if (this->kernel->kernelType == LINEAR) {
    this->theta =
        this->theta + temp1 * this->x.row(i1).t() + temp2 * this->x.row(i2).t();
  }
  // Update error cache using new Lagrange multipliers
  int exampleNum = this->ExampleNum();
  for (int i = 0; i < exampleNum; i++) {
    if (lagrangeMultiplier[i] > 0 && lagrangeMultiplier[i] < this->C) {
      this->errorCache[i] += temp1 * this->KernelCal(i1, i) +
                             temp2 * this->KernelCal(i2, i) - bDiff;
    }
  }

  this->errorCache[i1] = 0.0;
  this->errorCache[i2] = 0.0;

  // Store a1, a2 in the alpha array
  this->lagrangeMultiplier[i1] = a1;
  this->lagrangeMultiplier[i2] = a2;
  return 1;
}

int SmoSolver::ExamineExample(int i2) {
  double y2 = 0.0;
  double alpha2 = 0.0;
  double e2 = 0.0;
  double r2 = 0.0;

  alpha2 = this->lagrangeMultiplier[i2];
  y2 = this->y[i2];
  if (alpha2 > 0 && alpha2 < this->C) {
    e2 = this->errorCache[i2];
  } else {
    e2 = this->SvmOutputOnPoint(i2) - y2;
  }
  r2 = e2 * y2;

  int exampleNum = this->ExampleNum();
  //
  double tmax = 0.0;
  int i1 = 0;
  int k = 0;
  if ((r2 < -TOL && alpha2 < this->C) || (r2 > TOL && alpha2 > 0)) {
    for (i1 = -1, tmax = 0, k = 0; k < exampleNum; k++) {
      if (lagrangeMultiplier[k] > 0 && lagrangeMultiplier[k] < this->C) {
        double e1 = 0.0;
        double temp = 0.0;
        e1 = this->errorCache[k];
        temp = std::abs(e2 - e1);
        if (temp > tmax) {
          tmax = temp;
          i1 = k;
        }
      }
      if (i1 >= 0) {
        if (TakeStep(i1, i2)) {
          return 1;
        }
      }
    }

    for (int i = (int)(drand48() * exampleNum), k = i; k < exampleNum + i;
         k++) {
      i1 = k % exampleNum;
      if (lagrangeMultiplier[i1] > 0 && lagrangeMultiplier[i1] < C) {
        if (TakeStep(i1, i2)) {
          return 1;
        }
      }
    }
    for (int i = (int)(drand48() * exampleNum), k = i; k < exampleNum + i;
         k++) {
      i1 = k % exampleNum;
      if (TakeStep(i1, i2)) {
        return 1;
      }
    }
  }
  return 0;
}

int SmoSolver::ExampleNum() { return (int)this->x.n_rows; }

int SmoSolver::Train() {
  int exampleNum = this->ExampleNum();
  if (!trained) {
    this->b = 0.0;
    this->theta = zeros<vec>(this->x.n_cols);
    this->errorCache = zeros<vec>(exampleNum);
    this->lagrangeMultiplier = zeros<vec>(exampleNum);
  }

  unsigned int numChanged = 0;
  unsigned int examineAll = 1;
  while (numChanged > 0 || examineAll) {
    numChanged = 0;
    if (examineAll) {
      for (int i = 0; i < exampleNum; i++) {
        numChanged += ExamineExample(i);
      }
    } else {
      for (int i = 0; i < exampleNum; i++) {
        if (lagrangeMultiplier[i] != 0 && lagrangeMultiplier[i] != C) {
          numChanged += ExamineExample(i);
        }
      }
    }
    if (examineAll == 1) {
      examineAll = 0;
    } else if (numChanged == 0) {
      examineAll = 1;
    }
  }
  this->trained = true;
  return 0;
}
