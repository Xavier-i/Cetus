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

    this->y = new vec(y);
    this->m = m;
    this->trained = false;
}

LinearRegression::~LinearRegression(){
  delete x;
  delete y;
  delete w;
}

void AddData(double x[], double y[]){
  //To do
  delete this->w;
  this -> trained = false;
}
