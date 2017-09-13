#ifndef CETUS_INPUT_H_
#define CETUS_INPUT_H_
#include <armadillo>
enum ModelType { LINEARREGRESSION, LOGISTICREGRESSION, SVM }; /* model_type */

struct ModelInput {
  ModelType type;

  arma::mat data;
  arma::vec target;
};


#endif
