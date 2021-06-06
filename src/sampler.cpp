#include "utils.h"
#include "RcppArmadillo.h"


// [[Rcpp::export]]
Rcpp::List sampler(arma::uvec Y_sorted, arma::uvec Y_failure, arma::mat X, 
	double lambda, double a_0, double b_0, double kernel_sd, 
	arma::uword mcmc_samples, bool verbose, double e, int L)
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
	w = R::rbeta(a_0, b_0);

	// update p
	for (arma::uword j = 0; j < p; ++j) {
	    z(j) = 0;
	    double p0 = log(1-w) + log_PL(X, b % z, Y_sorted, Y_failure); 
	    z(j) = 1;
	    double p1 = log(w) + log_PL(X, b % z, Y_sorted, Y_failure);
	    
	    // compute the conditional posterior for z(i) and sample from it
	    double prob0 = sigmoid(p0 - p1);
	    double prob1 = sigmoid(p1 - p0);
	    double a = prob1 / (prob0 + prob1);
	    z(j) = static_cast<double>(a > R::runif(0, 1));
	}

	// update b
	// for (arma::uword j = 0; j < p; ++j) {
	//     double b_old = b(j);
	//     double b_new = R::rnorm(b_old, kernel_sd * pow(10.0 / lambda, (1.0 - z(j))));
	//     
	//     // MH denominator
	//     double den = log_PL(X, b % z, Y_sorted, Y_failure) +
	// 		log_Laplace(b_old, lambda);
	//     
	//     // MH numerator
	//     b(j) = b_new;
	//     double num = log_PL(X, b % z, Y_sorted, Y_failure) +
	// 		log_Laplace(b_new, lambda);

	//     double a = std::min(1.0, exp(num - den));
	//     if (a <= R::runif(0, 1)) b(j) = b_old; 
	// }

	// update b
	arma::vec r = arma::vec(p, arma::fill::randn);
	arma::vec r_old = r;
	arma::vec b_new = b % z;
	for (int l = 0; l < L; ++l) {
	    leapfrog(b_new, r, e, X, Y_sorted, Y_failure, lambda);
	}
	double num = log_PL(X, b_new, Y_sorted, Y_failure) - 0.5 * dot(r, r);
	double den = log_PL(X, b, Y_sorted, Y_failure) - 0.5 * dot(r_old, r_old);
	double a = exp(num - den);
	if (a > R::runif(0, 1)) b = b_new;
	
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


