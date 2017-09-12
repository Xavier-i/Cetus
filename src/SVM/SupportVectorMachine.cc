#include "Kernel.h"
#include "Para.h"
#include "Solver.h"
#include "SupportVectorMachine.h"
#include <armadillo>
#include <assert.h>
#include <iostream>

using namespace arma;

SupportVectorMachine::SupportVectorMachine(mat x, vec y, SvmParameter *para)
    : x{x}, y{y}, trained{false} {
  assert(x.n_rows == y.n_rows);

  // Create bias column and append at the end of  x
  mat bias = ones<mat>(this->ExampleNumber(), 1);
  this->x.insert_cols(0, bias);
  this->kernel = new Kernel(para);
  this->solver = new SmoSolver(this->x, y, kernel, para->regParameterC);
}

SupportVectorMachine::~SupportVectorMachine() {
  delete this->kernel;
  delete this->solver;
}

uword SupportVectorMachine::ExampleNumber() { return this->x.n_rows; }

int SupportVectorMachine::Train() {
  trained = true;
  return this->solver->Train();
}

// SVM doesn't return probablity
int SupportVectorMachine::Predict(vec x) {
  if (!this->trained) {
    std::cerr << "This model hasn't been trained" << std::endl;
    return 0.0;
  }
  vec bias = vec("1");
  vec input = x;
  input.insert_rows(0, bias);
  if (this->solver->Predict(input) >= 0) {
    return 1;
  } else {
    return -1;
  }
}
