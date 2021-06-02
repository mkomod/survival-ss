#include "utils.h"
#include "RcppArmadillo.h"


// [[Rcpp::export]]
Rcpp::List sampler(arma::uvec Y_sorted, arma::uvec Y_failure, arma::mat X, 
	double lambda, double kernel_sd, arma::uword mcmc_samples, bool verbose)
{
    arma::uword p = X.n_cols;
    arma::uword n = X.n_rows;
    
    // save values
    arma::mat Z = arma::mat(p, mcmc_samples, arma::fill::zeros);
    arma::mat B = arma::mat(p, mcmc_samples, arma::fill::zeros);
    arma::mat W = arma::mat(1, mcmc_samples, arma::fill::zeros);

    // initialise
    double w = R::runif(0, 1);
    arma::vec b = arma::vec(p, arma::fill::randn);
    arma::vec z = arma::vec(p, arma::fill::randu);
    z = arma::round(z);

    if (verbose)
	Rcpp::Rcout << "\n Running sampler    [";

    for (arma::uword iter = 0; iter < mcmc_samples; ++iter) {
	// update w
	w = R::runif(0, 1);

	// update p
	for (arma::uword j = 0; j < p; ++j) {
	    z(j) = 0;
	    double p0 = log(w) + log_PL(X, b % z, Y_sorted, Y_failure); 
	    z(j) = 1;
	    double p1 = log(1-w) + log_PL(X, b % z, Y_sorted, Y_failure);
	    
	    // compute the conditional posterior for z(i) and sample from it
	    double prob0 = sigmoid(p0 - p1);
	    double prob1 = sigmoid(p1 - p0);
	    double a = prob1 / (prob0 + prob1);
	    z(j) = static_cast<double>(a > R::runif(0, 1));
	}

	// update b
	for (arma::uword j = 0; j < p; ++j) {
	    double b_old = b(j);
	    double b_new = R::rnorm(b_old, kernel_sd * pow(10.0 / lambda, (1.0 - z(j))));
	    
	    // MH denominator
	    double p0 = log_PL(X, b % z, Y_sorted, Y_failure) +
			log_Laplace(b_old, lambda);
	    
	    // MH numerator
	    b(j) = b_new;
	    double p1 = log_PL(X, b % z, Y_sorted, Y_failure) +
			log_Laplace(b_new, lambda);

	    double a = std::min(1.0, exp(p1 - p0));
	    if (a <= R::runif(0, 1)) b(j) = b_old; 
	}
	
	W(iter) = w;
	B.col(iter) = b;
	Z.col(iter) = z;

	if (verbose && (iter % (mcmc_samples/50) == 0))
	    Rcpp::Rcout << "#";

	Rcpp::checkUserInterrupt();
    }

    if (verbose)
	Rcpp::Rcout << "]\n\n";

    return Rcpp::List::create(
	Rcpp::Named("w") = W,
	Rcpp::Named("b") = B,
	Rcpp::Named("z") = Z
    );
}


