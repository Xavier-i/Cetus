#include "Kernel.h"
#include "Solver.h"
#include "SupportVectorMachine.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

SupportVectorMahchine::SupportVectorMahchine(mat &x, vec &y, double regParaC,
                                             KernelType type)
    : x{x}, y{y}, trained{false} {
  assert(x.n_rows == y.n_rows);

  // Create bias column and append at the end of  x
  mat bias = ones<mat>(this->ExampleNumber(), 1);
  this->x.insert_cols(0, bias);
  this->kernel = new Kernel(type);
  this->solver = new SmoSolver(x, y, kernel);
}

SupportVectorMahchine::~SupportVectorMahchine() {
  delete this->kernel;
  delete this->solver;
}

uword LogisticRegression::ExampleNumber() { return this->x.n_rows; }

int SupportVectorMachine::Train(){
  this->solver.train();
  trained=true;
}
// SVM doesn't return probablity
int SupportVectorMachine::Predict(vec &x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  vec bias = vec("1");
  vec input = x;
  input.insert_rows(0, bias);
  if (this->solver->predict(input) >= 0) {
    return 1;
  } else {
    return 0;
  }
}
