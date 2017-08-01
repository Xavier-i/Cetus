#include <iostream>
#include "LinearRegression.h"
#include <armadillo>

using namespace std;
using namespace arma;

LinearRegression::LinearRegression(vector< vector<double> > x, vector<double> y, int m) {
    /*still in test
     row and column may switch
     Using mat(ptr_aux_mem, n_rows, n_cols, copy_aux_mem, strict)
    */

    this->x = new mat(x,m,x[0].size(),true, false);
//
// add bias column to x
//
    this->y = new vec(y);
    this->m = m;
    this->trained = false;
}

LinearRegression::~LinearRegression(){
  delete x;
  delete y;
  delete w;
}

void LinearRegression::AddData(double x[], double y[]){
  //To do
  delete this->w;
  this -> trained = false;
}

void LinearRegression::train(){
  mat xtx = (this->x->t() * (*this->x));
  // Check if xtx is full-rank matrix
  if ( rank(xtx) == this->m ){
    this->w = new mat(inv(xtx) * this->x->t() * y);
    this->trained = true;
  } else{
    std::cerr << "you have to regularize your data set"<< std::endl;
  }
}

double LinearRegression::predict(vector<double> x){
  if (!this->trained){
    std::cerr<<"This model hasn't been trained"<<std::endl;
    return 0.0;
  }else{
    return ((*this->w)*vec(x)).eval()(0,0);
  }

}
