#include "LinearRegression.h"
#include "LogisticRegression.h"
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
  std::cout << A->SelfCost() << std::endl;
}

void test3() {
  mat x = mat("2 2; 3 3; 4 4; -1 2; -10 8; 8 9; -4 -4;1 -3; 8 1; 3 -4; 1 -1; "
              "-1 1; 4 -6");
  vec y = vec("1 1 1 0 0 1 0 0 1 0 0 0 0 ");
  LogisticRegression *A = new LogisticRegression(x, y);
  A->Train(gradientDescent, 0.1, 100000);
  vec p = vec("3 0");
  vec q = vec("5 -5");
  vec f = vec("3 7");
  std::cout << A->Predict(p) << std::endl;
  std::cout << A->Predict(q) << std::endl;
  std::cout << A->Predict(f) << std::endl;
  std::cout << A->SelfCost() << std::endl;
}
int main() {
  testFailed();
  testSucc();
  testSucc2();
  test3();
}
