#include "LogisticRegression.h"
#include <armadillo>
#include <assert.h>
#include <iostream>
#include <math.h>

using namespace arma;

LogisticRegression::LogisticRegression(mat &x, vec &y)
    : x{x}, y{y}, trained{false} {
  assert(x.n_rows == y.n_rows);

  // Create bias column and append at the end of  x
  mat bias = ones<mat>(this->ExampleNumber(), 1);
  this->x.insert_cols(0, bias);
}

LogisticRegression::~LogisticRegression() {}

void LogisticRegression::AddData(mat &extraX, vec &extraY) {
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

void LogisticRegression::Train(TrainingType Type, double alpha,
                               unsigned int iters) {
  /*if (Type == normalEquation) {
    this->NormalEquation();
  } else*/
  if (Type == gradientDescent) {
    this->GradientDescent(alpha, iters);
  } else {
    std::cerr << "Invalid training type" << std::endl;
  }
}

uword LogisticRegression::ExampleNumber() { return this->x.n_rows; }

double LogisticRegression::Predict(vec &x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  vec bias = vec("1");
  vec input = x;
  input.insert_rows(0, bias);
  double probablity = (input.t() * this->theta).eval()(0, 0);
  if (probablity >= probabilityThreshold) {
    return 1;
  } else {
    return 0;
  }
}

arma::mat LogisticRegression::SigmoidFunction(arma::mat inputX) {
  return 1 / (1 + exp(-inputX));
}

/*
float LogisticRegression::CostFunction() {
  vec h = this->SigmoidFunction(this->x * this->theta);
  vec ve = - this->y.t() * log(h) - ((1 - y).t() * log(1 - h));
  return (1 / (float)this->ExampleNumber() * ve).eval()(0, 0);
}*/

vec LogisticRegression::CostDerivative() {
  vec h = this->SigmoidFunction(this->x * this->theta);
  vec deriv = this->x.t() * (h - this->y);
  return 1 / (double)this->ExampleNumber() * deriv;
}

void LogisticRegression::GradientDescent(double alpha, unsigned int iters) {
  if (this->trained != true || this->theta.n_rows != this->x.n_cols) {
    // Initialize Theta
    this->theta = zeros<vec>(this->x.n_cols);
  }
  for (unsigned int i = 0; i < iters; i++) {
    this->theta = this->theta - (alpha * this->CostDerivative());
  }

  this->trained = true;
}
