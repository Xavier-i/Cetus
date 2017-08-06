#include "LinearRegression.h"
#include <armadillo>
#include <iostream>

LinearRegression::LinearRegression(arma::mat *x, arma::vec *y, arma::uword m)
    : x{x}, y{y}, m{m}, trained{false} {}

LinearRegression::~LinearRegression() {
  delete x;
  delete y;
  delete w;
}

void LinearRegression::AddData(double x[], double y[]) {
  // TODO:
  delete this->w;
  this->trained = false;
}

void LinearRegression::Train() {
  arma::mat xtx = (this->x->t() * (*this->x));
  // Check if xtx is full-rank matrix
  if (rank(xtx) == this->m) {
    this->w = new arma::mat(inv(xtx) * this->x->t() * *y);
    this->trained = true;
  } else {
    std::cerr << "you have to regularize your data set" << std::endl;
  }
}

double LinearRegression::Predict(arma::vec *x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  return ((*this->w) * (*x)).eval()(0, 0);
}
