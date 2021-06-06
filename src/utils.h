#ifndef UTILS_H
#define UTILS_H

#include "RcppArmadillo.h"


double log_PL(arma::mat X, arma::vec b, arma::uvec Y_sorted, arma::uvec Y_failure);
double log_Laplace(double beta, double lambda);
double sigmoid(double x);
arma::rowvec log_PL_grad(const arma::mat &X, const arma::vec &b, 
	const arma::uvec &Y_sorted, const arma::uvec &Y_failure);
arma::rowvec log_Laplace_grad(arma::vec beta, double lambda);
void leapfrog(arma::vec &b, arma::vec &r, double e, const arma::mat &X,
	const arma::uvec &Y_sorted, const arma::uvec &Y_failure, double lambda);


#endif
