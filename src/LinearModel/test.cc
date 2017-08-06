#include "LinearRegression.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

main() {
  mat x = mat("1 0 0; 0 0 1");
  vec y = vec("0 1");
  LinearRegression *A = new LinearRegression(x, y);
  A->Train();
  vec p = vec("1 0 0");
  std::cout << A->Predict(p) << std::endl;
}
