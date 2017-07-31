#ifndef MODEL_LINEARREGRESSION_H_
#define MODEL_LINEARREGRESSION_H_
#include <armadillo>
#include <vector>

using namespave std;
using namespace arma;

class LinearRegression {

  public:
    // First feature
    mat *x;

    // Target feature
    vec *y;

    // Number of training examples
    int m;

    // Model Trained or not
    bool trained;

    // Create a new instance from the given data set.
    LinearRegression(vector< vector<double> > x, vector<double> y, int m);

    // Add other features
    void AddData(double x[], double y[])

    // Train the model
    void train();

    /**
     * Try to predict y, given an x.
     */
    double predict(double x);

    //Destructor
private:
    mat *w
    /**
     * Compute the cost J.
     */
    static double compute_cost(double x[], double y[], double theta[], int m);

    /**
     * Compute the hypothesis.
     */
    static double h(double x, double theta[]);

    /**
     * Calculate the target feature from the other ones.
     */
    static double *calculate_predictions(double x[], double theta[], int m);

    /**
     * Performs gradient descent to learn theta by taking num_items gradient steps with learning rate alpha.
     */
    static double *gradient_descent(double x[], double y[], double alpha, int iters, double *J, int m);

};

#endif
