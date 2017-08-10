#include "SupportVectorMahchine.h"

#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

SupportVectorMahchine::SupportVectorMahchine(mat &x, vec &y, double regPara)
    : x{x}, y{y}, trained{false}, regPara{regPara} {
  assert(x.n_rows == y.n_rows);

  // Create bias column and append at the end of  x
  mat bias = ones<mat>(this->ExampleNumber(), 1);
  this->x.insert_cols(0, bias);
}

SupportVectorMahchine::~SupportVectorMahchine() {}
