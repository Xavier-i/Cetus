#ifndef MODEL_SUPPORTVECTORMACHINE_H_
#define MODEL_SUPPORTVECTORMACHINE_H_
#include "Kernel.h"
#include "Para.h"
#include "Solver.h"
#include <armadillo>

class SupportVectorMachine {
  // First feature
  arma::mat x;

  // Target feature
  // Elements in y have to be either 1 or 0
  arma::vec y;
  SmoSolver *solver;
  // Kernel
  Kernel *kernel;

public:
  // Model Trained or not
  bool trained;

  // Create a new instance from the given data set.
  SupportVectorMachine(arma::mat x, arma::vec y, double regParaC,
                       SvmParameter *para);

  // Destructor
  ~SupportVectorMachine();
  /*
    // Add other features
    void AddData(arma::mat &extraX, arma::vec &extraY);

    // Train the model
    void Train(TrainingType Type, double alpha = 0, unsigned int iters = 0);
*/
  // Return number of example
  arma::uword ExampleNumber();

  // SVM doesn't return probablity
  // Predict y according to given x
  int Predict(arma::vec x);
  int Train();
};

#endif
