#ifndef MODEL_LOGISTICREGRESSION_H_
#define MODEL_LOGISTICREGRESSION_H_
#include "TrainingType.h"
#include <armadillo>

class LogisticRegression {
  // First feature
  arma::mat x;

  // Target feature
  // Elements in y have to be either 1 or 0
  arma::vec y;

  // Vector for predication
  arma::vec theta;

public:
  // Regularization rate
  double regPara;

  double probabilityThreshold = 0.5;

  // Model Trained or not
  bool trained;

  // Create a new instance from the given data set.
  LogisticRegression(arma::mat x, arma::vec y, double regPara = 0);

  // Destructor
  ~LogisticRegression();

  // Add other features
  void AddData(arma::mat extraX, arma::vec extraY);

  // Train the model
  void Train(TrainingType Type, double alpha = 0, unsigned int iters = 0);

  // Return number of example
  arma::uword ExampleNumber();

  // Predict y according to given x
  double Predict(arma::vec x);

  // Predict probablity of 1
  double Probablity(arma::vec x);

  // Cost function using the own data;
  double SelfCost();

  // Cost Function
  // May return -nan when Cost is really small
  double Cost(arma::mat inputX);

private:
  // Initialize Theta if doesn't exist.
  void InitializeTheta();

  // Compute Cost Functions's Derivative
  arma::vec CostDerivative();

  arma::mat SigmoidFunction(arma::mat inputX);

  // Normal Equation Method to find theta
  // void NormalEquation();

  // Performs gradient descent to learn theta by taking iters gradient steps
  //   with learning rate alpha.
  void GradientDescent(double alpha, unsigned int iters);
};

#endif
