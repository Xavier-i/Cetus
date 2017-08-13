#include "Kernel.h"
#ifndef MODEL_SUPPORTVECTORMACHINE_H_
#define MODEL_SUPPORTVECTORMACHINE_H_
#include <armadillo>

class SupportVectorMahchine {
  // First feature
  arma::mat x;

  // Target feature
  // Elements in y have to be either 1 or 0
  arma::vec y;

  // Vector for predication
  arma::vec theta;

public:
  // Regularization rate
  //
  // Small C -> Large Margin, insensitive to outlier
  double regParaC;

  // Model Trained or not
  bool trained;

  // Create a new instance from the given data set.
  SupportVectorMahchine(arma::mat &x, arma::vec &y, double regParaC = 1,
                        KernelType kernel = LINEAR);

  // Destructor
  ~SupportVectorMahchine();
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
  double Predict(arma::vec &x);

  // Cost function using the own data;
  double SelfCost();

  // Cost Function
  // May return -nan when Cost is really small
  double Cost(arma::mat &inputX);

private:
  // Initialize Theta if doesn't exist.
  void InitializeTheta();
  /*
      // Compute Cost Functions's Derivative
      arma::vec CostDerivative();


      // Performs gradient descent to learn theta by taking iters gradient steps
      //   with learning rate alpha.
      void GradientDescent(double alpha, unsigned int iters);*/
};

#endif
