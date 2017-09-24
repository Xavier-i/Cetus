#ifndef CETUS_INPUT_H_
#define CETUS_INPUT_H_
#include <armadillo>
#include <string>

enum ModelType { LINEARREGRESSION, LOGISTICREGRESSION, SVM }; /* model_type */
enum FileType { CSV };
enum Object { TARGET, DATA };

struct ModelInput {
  ModelInput(ModelType modelType) : type{modelType} {}
  ~ModelInput() {}

  // Load From File
  //bool LoadFromFile(std::string path, FileType inputType, Object goal);

  //*ModelInput

  ModelType type;

  arma::mat data;
  arma::vec target;
};

#endif
