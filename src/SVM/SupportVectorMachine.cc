#include "Kernel.h"
#include "SupportVectorMahchine.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

SupportVectorMahchine::SupportVectorMahchine(mat &x, vec &y, double regParaC,
                                             KernelType kernel)
    : x{x}, y{y}, trained{false}, regParaC{regParaC} {
  assert(x.n_rows == y.n_rows);

  // Create bias column and append at the end of  x
  mat bias = ones<mat>(this->ExampleNumber(), 1);
  this->x.insert_cols(0, bias);
}

SupportVectorMahchine::~SupportVectorMahchine() {}

uword LogisticRegression::ExampleNumber() { return this->x.n_rows; }

// SVM doesn't return probablity
double SupportVectorMachine::Predict(vec &x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  vec bias = vec("1");
  vec input = x;
  input.insert_rows(0, bias);
  if ((input.t() * this->theta).eval()(0, 0) >= 0) {
    return 1;
  } else {
    return 0;
  }
}

double SupportVectorMachine::SelfCost() { return this->Cost(this->x); }

double SupportVectorMachine::Cost(mat &inputX) {
  this->InitializeTheta();
  //--                    h = g(X Theta)                   --//
  /*--J(Theta) = C * (-y^T Cost1(Theta X) - (1-y)^T Cost0(Theta X) +1/2
  Theta^2--*/
  assert(inputX.n_cols == this->theta.n_rows);
  vec thetaX = inputX * this->theta;
  // cost1 cost function when y = 1
  // cost0 cost function when y = 0
  vec A = (this->y.t() * (this->cost1(thetaX)) + ((1 - y).t() * (this->cost0(thetaX));
  //vec thetaWithoutFirst = this->theta;
  //thetaWithoutFirst[0] = 0;
  vec B = (double)1/  2 *
      this->theta.t() * this->theta);
  return (this->regParaC*A + B).eval()(0, 0);
}

void SupportVectorMachine::InitializeTheta() {
  if (this->trained != true || this->theta.n_rows != this->x.n_cols) {
    // Initialize Theta
    this->theta = zeros<vec>(this->x.n_cols);
  }
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
