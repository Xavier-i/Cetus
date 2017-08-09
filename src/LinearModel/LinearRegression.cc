#include "LinearRegression.h"
#include "TrainingType.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

LinearRegression::LinearRegression(mat &x, vec &y)
    : x{x}, y{y}, trained{false} {
  assert(x.n_rows == y.n_rows);

  // Create bias column and append at the end of  x
  mat bias = ones<mat>(this->ExampleNumber(), 1);
  this->x.insert_cols(0, bias);
}

LinearRegression::~LinearRegression() {}

void LinearRegression::AddData(mat &extraX, vec &extraY) {
  assert(extraX.n_rows == extraY.n_rows);
  // Add 1 because x has a bias column
  assert((extraX.n_cols + 1) == this->x.n_cols);

  this->trained = false;
  // Add Bias column to latest added input
  mat bias = ones<mat>(extraX.n_rows, 1);
  mat inputX = extraX;
  inputX.insert_cols(0, bias);
  this->x.insert_rows(this->x.n_rows, inputX);
  this->y.insert_rows(this->y.n_rows, extraY);
}

void LinearRegression::Train(TrainingType Type, double alpha,
                             unsigned int iters) {
  if (Type == normalEquation) {
    this->NormalEquation();
  } else if (Type == gradientDescent) {
    this->GradientDescent(alpha, iters);
  } else {
    std::cerr << "Invalid training type" << std::endl;
  }
}

void LinearRegression::NormalEquation() {
  mat xtx = (this->x.t() * this->x);
  // Check if xtx is full-rank matrix
  if (rank(xtx) == xtx.n_rows) {
    this->theta = pinv(xtx) * this->x.t() * this->y;
    this->trained = true;
  } else {
    std::cerr << "you have to regularize your data set" << std::endl;
  }
}

uword LinearRegression::ExampleNumber() { return this->x.n_rows; }

double LinearRegression::Predict(vec &x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  vec bias = vec("1");
  vec input = x;
  input.insert_rows(0, bias);
  return (input.t() * this->theta).eval()(0, 0);
}

vec LinearRegression::CostDerivative() {
  vec deriv = (((this->x * this->theta) - this->y).t() * this->x).t();
  return 1 / (double)this->ExampleNumber() * deriv;
}

double LinearRegression::SelfCost() { return this->Cost(this->x); }

double LinearRegression::Cost(mat &inputX) {
  this->InitializeTheta();
  //--J(Theta) = 1/2m * (X Theta - y)^T (X Theta - y)--//
  assert(inputX.n_cols == this->theta.n_rows);
  vec ve = (inputX * this->theta) - this->y;
  return (((float)1 / 2) * this->ExampleNumber() * ve.t() * ve).eval()(0, 0);
}

void LinearRegression::GradientDescent(double alpha, unsigned int iters) {
  this->InitializeTheta();
  for (unsigned int i = 0; i < iters; i++) {
    this->theta = this->theta - (alpha * this->CostDerivative());
  }

  this->trained = true;
}

void LinearRegression::InitializeTheta() {
  if (this->trained != true || this->theta.n_rows != this->x.n_cols) {
    // Initialize Theta
    this->theta = zeros<vec>(this->x.n_cols);
  }
}
