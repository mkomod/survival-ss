#ifndef UTILS_H
#define UTILS_H

#include "RcppArmadillo.h"

// log partial likelihood and jacobian
double log_PL(arma::mat X, arma::vec b, arma::uvec Y_sorted, arma::uvec Y_failure);
arma::rowvec log_PL_grad(const arma::mat &X, const arma::vec &b, 
	const arma::uvec &Y_sorted, const arma::uvec &Y_failure);

// log laplace and jacobian
double log_Laplace(double beta, double lambda);
arma::rowvec log_Laplace_grad(arma::vec beta, double lambda);

// misc
void leapfrog(arma::vec &b, arma::vec &r, const arma::vec &e, const arma::mat &X,
	const arma::uvec &Y_sorted, const arma::uvec &Y_failure, double lambda);
double sigmoid(double x);


#endif
