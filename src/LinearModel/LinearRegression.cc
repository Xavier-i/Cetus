#include "LinearRegression.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

LinearRegression::LinearRegression(mat &x, vec &y)
    : x{x}, y{y}, trained{false} {
  assert(x.n_rows == y.n_rows);

  // Create Bias Layer and append at the end of  x
  mat bias = ones<mat>(y.n_rows, 2);
  this->x.insert_cols(x.n_cols, bias);
}

LinearRegression::~LinearRegression() {}

void LinearRegression::AddData(mat &extraX, vec &extraY) {
  assert(extraX.n_rows == extraY.n_rows);
  assert(extraX.n_rows == this->x.n_rows);
  this->trained = false;
  this->x.insert_rows(this->x.n_rows, extraX);
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
  return (this->w * x).eval()(0, 0);
}
