#include "LogisticRegression.h"
#include "TrainingType.h"
#include <armadillo>
#include <assert.h>
#include <iostream>
#include <math.h>

using namespace arma;

LogisticRegression::LogisticRegression(mat x, vec y, double regPara)
    : x{x}, y{y}, regPara{regPara}, trained{false} {
  assert(x.n_rows == y.n_rows);

  // Create bias column and append at the end of  x
  mat bias = ones<mat>(this->ExampleNumber(), 1);
  this->x.insert_cols(0, bias);
}

LogisticRegression::~LogisticRegression() {}

void LogisticRegression::AddData(mat extraX, vec extraY) {
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

double LogisticRegression::Probablity(vec x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  vec bias = vec("1");
  vec input = x;
  input.insert_rows(0, bias);
  return dot(input, this->theta);
}

double LogisticRegression::Predict(vec x) {
  double prob = this->Probablity(x);
  if (prob >= probabilityThreshold) {
    return 1;
  } else {
    return 0;
  }
}

arma::mat LogisticRegression::SigmoidFunction(arma::mat inputX) {
  //--     1      --//
  //-- ---------- --//
  //-- 1 + e^(-inpuX) --//
  return 1 / (1 + exp(-inputX));
}

double LogisticRegression::SelfCost() { return this->Cost(this->x); }

double LogisticRegression::Cost(mat inputX) {
  this->InitializeTheta();
  //--                    h = g(X Theta)                   --//
  //--J(Theta) = 1/m * (-y^T log(h) - (1-y)^T log(1-h)) +lambda/2m theta^2--//
  assert(inputX.n_cols == this->theta.n_rows);
  vec h = this->SigmoidFunction(inputX * this->theta);
  vec ve = (-this->y.t() * log(h)) - ((1 - y).t() * log(1 - h));
  vec thetaWithoutFirst = this->theta;
  thetaWithoutFirst[0] = 0;

  return (1 / (float)this->ExampleNumber() * ve +
          this->regPara / (double)this->ExampleNumber() * 2 *
              thetaWithoutFirst.t() * thetaWithoutFirst)
      .eval()(0, 0);
}

vec LogisticRegression::CostDerivative() {
  vec h = this->SigmoidFunction(this->x * this->theta);
  vec deriv = this->x.t() * (h - this->y);
  vec thetaWithoutFirst = this->theta;
  thetaWithoutFirst[0] = 0;
  return 1 / (double)this->ExampleNumber() * deriv +
         this->regPara / (double)this->ExampleNumber() * thetaWithoutFirst;
}

void LogisticRegression::GradientDescent(double alpha, unsigned int iters) {
  this->InitializeTheta();
  for (unsigned int i = 0; i < iters; i++) {
    this->theta = this->theta - (alpha * this->CostDerivative());
  }

  this->trained = true;
}

void LogisticRegression::InitializeTheta() {
  if (this->trained != true || this->theta.n_rows != this->x.n_cols) {
    // Initialize Theta
    this->theta = zeros<vec>(this->x.n_cols);
  }
}
