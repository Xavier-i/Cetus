#include "LinearRegression.h"
#include <armadillo>
#include <iostream>
#include <assert.h>

using namespace arma;

LinearRegression::LinearRegression(mat *x, vec *y, uword m)
    : x{x}, y{y}, m{m}, trained{false} {
      assert(x->n_rows == y->n_rows);
      assert(y->n_rows == m);

      //Create Bias Layer and append at the end of  x
      mat bias = ones<mat>(m,2);
      this->x->insert_cols(x->n_cols, bias);
    }

LinearRegression::~LinearRegression() {
  delete x;
  delete y;
  delete w;
}

void LinearRegression::AddData(mat *extraX, vec *extraY, uword m){
  assert(extraX->n_rows == extraY->n_rows);
  assert(extraY->n_rows == m);

  delete this->w;
  this->trained = false;
  this->x->insert_rows(this->x->n_rows,*extraX);
  this->y->insert_rows(this->y->n_rows,*extraY);
  this->m +=m;
}

void LinearRegression::Train() {
  mat xtx = (this->x->t() * (*this->x));
  // Check if xtx is full-rank matrix
  if (rank(xtx) == this->m) {
    this->w = new mat(inv(xtx) * this->x->t() * *y);
    this->trained = true;
  } else {
    std::cerr << "you have to regularize your data set" << std::endl;
  }
}

double LinearRegression::Predict(vec *x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  return ((*this->w) * (*x)).eval()(0, 0);
}
