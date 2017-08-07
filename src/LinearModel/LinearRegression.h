#ifndef MODEL_LINEARREGRESSION_H_
#define MODEL_LINEARREGRESSION_H_
#include <armadillo>

class LinearRegression {
  // First feature
  arma::mat x;

  // Target feature
  arma::vec y;

  // Matrix for predication
  arma::mat w;

public:
  // Model Trained or not
  bool trained;

  // Create a new instance from the given data set.
  LinearRegression(arma::mat &x, arma::vec &y);

  // Destructor
  ~LinearRegression();

  // Add other features
  void AddData(arma::mat &extraX, arma::vec &extraY);

  // Train the model
  void Train();

  // Return number of example
  arma::uword ExampleNumber();

  // Predict y according to given x
  double Predict(arma::vec &x);

private:
  /**
   * Compute the cost J.
   */
  // static double compute_cost(double x[], double y[], double theta[], int m);

  /**
   * Compute the hypothesis.
   */
  // static double h(double x, double theta[]);

  /**
   * Calculate the target feature from the other ones.
   */
  // static double *calculate_predictions(double x[], double theta[], int m);

  /**
   * Performs gradient descent to learn theta by taking num_items gradient steps
   * with learning rate alpha.
   */
  // static double *gradient_descent(double x[], double y[], double alpha, int
  // iters, double *J, int m);
};

#endif
