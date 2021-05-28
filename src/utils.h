#ifndef UTILS_H
#define UTILS_H

#include "RcppArmadillo.h"


double log_PL(arma::mat X, arma::vec b, arma::uvec Y_sorted, arma::uvec Y_failure);
double log_Laplace(double beta, double lambda);
double sigmoid(double x);


#endif
