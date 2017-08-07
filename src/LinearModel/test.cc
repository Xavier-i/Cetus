#include "LinearRegression.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

void testFailed() {
  mat x = mat("1 0 0; 0 0 1");
  vec y = vec("0 1");
  LinearRegression *A = new LinearRegression(x, y);
  A->Train(normalEquation);
  vec p = vec("1 0 0");
}

void testSucc() {
  mat x = mat("3 0; 0 4;2 5; 5 2");
  vec y = vec("3 8 12 9 ");
  LinearRegression *A = new LinearRegression(x, y);
  A->Train(normalEquation);
  vec p = vec("3 0");
  std::cout << A->Predict(p) << std::endl;
  vec q = vec("0 8");
  std::cout << A->Predict(q) << std::endl;
  mat ex = mat("0 0;");
  vec ey = vec("0");

  A->AddData(ex, ey);
  A->Train(normalEquation);
  std::cout << A->Predict(q) << std::endl;
}

void testSucc2() {
  mat x = mat("3 0; 0 4;2 5;0 0; 5 2; 7 0; 0 8");
  vec y = vec("3 8 12 0 9 7 16");
  LinearRegression *A = new LinearRegression(x, y);
  A->Train(gradientDescent, 0.1, 10000);
  vec p = vec("3 0");
  std::cout << A->Predict(p) << std::endl;
  vec q = vec("0 8");
  std::cout << A->Predict(q) << std::endl;
  mat ex = mat("0 0;");
  vec ey = vec("0");

  A->AddData(ex, ey);
  A->Train(gradientDescent, 0.01, 1000);
  std::cout << A->Predict(q) << std::endl;
}

int main() {
  testFailed();
  testSucc();
  testSucc2();
}
