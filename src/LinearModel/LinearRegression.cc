#include "LinearRegression.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

LinearRegression::LinearRegression(mat &x, vec &y) : trained{false} {
  assert(x.n_rows == y.n_rows);

  // Create Bias Layer and append at the end of  x
  mat bias = ones<mat>(x.n_rows, 1);
  mat inputX = x;
  inputX.insert_cols(x.n_cols, bias);
  this->x = inputX;
  vec inputY = y;
  this->y = inputY;
  // this->x = inputX;
}

LinearRegression::~LinearRegression() {}

void LinearRegression::AddData(mat &extraX, vec &extraY) {
  assert(extraX.n_rows == extraY.n_rows);
  assert((extraX.n_cols + 1) == this->x.n_cols);

  this->trained = false;
  // Add Bias Layer to latest added input
  mat bias = ones<mat>(extraX.n_rows, 1);
  mat inputX = extraX;
  inputX.insert_cols(inputX.n_cols, bias);
  this->x.insert_rows(this->x.n_rows, inputX);
  this->y.insert_rows(this->y.n_rows, extraY);
}

void LinearRegression::Train() {
  mat xtx = (this->x.t() * this->x);
  // Check if xtx is full-rank matrix
  if (rank(xtx) == xtx.n_rows) {
    this->w = inv(xtx) * this->x.t() * y;
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
  input.insert_rows(x.n_rows, bias);
  return (input.t() * this->w).eval()(0, 0);
}
