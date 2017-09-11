#include "SupportVectorMachine.h"
#include "Para.h"
#include <armadillo>
#include <assert.h>
#include <iostream>
using namespace std;
using namespace arma;

void test1() {
  mat x = mat("2 2; 3 3; 4 4; -1 2; -10 8; 8 9; -4 -4;1 -3; 8 1; 3 -4; 1 -1; "
              "-1 1; 4 -6");
  vec y = vec("1 1 1 -1 -1 1 -1 -1 1 -1 -1 -1 -1 ");
  SvmParameter *para = new SvmParameter();
  SupportVectorMachine *A = new SupportVectorMachine(x, y, 1.0, para);
  A->Train();

  vec p = vec("3 0");
  vec q = vec("5 -5");
  vec f = vec("3 7");
  std::cout << A->Predict(p) << std::endl;
  std::cout << A->Predict(q) << std::endl;
  std::cout << A->Predict(f) << std::endl;
}

int main() { test1(); }
