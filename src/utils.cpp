#include "utils.h"


// Evaluate the log partial likelihood
// [[Rcpp::export]]
double log_PL(arma::mat X, arma::vec b, arma::uvec Y_sorted, 
	arma::uvec Y_failure)
{
    arma::vec xb = X * b;
    double a = max(xb);

    double denom = 0.0;
    double tot = 0.0;
    int ind_last_failure = X.n_rows;
    int ind_curr_failure = 0;

    for (int i = (Y_failure.n_rows - 1); i >= 0; --i) {
	ind_curr_failure = Y_failure(i);
	auto R = arma::span(ind_curr_failure, ind_last_failure - 1);

	// compute the denom and num
	denom += sum(exp(xb(Y_sorted(R)) - a));
	tot += xb(Y_sorted(ind_curr_failure)) - (a + log(denom));

	ind_last_failure = ind_curr_failure;
    }
    return tot;
}


// Univariate log laplace density
double log_Laplace(double beta, double lambda)
{
    return log(lambda) - log(2.0) - lambda * std::abs(beta);
}


double sigmoid(double x)
{
    double res = 0.0;
    if (x >= 0) {
	res = 1.0 / (1.0 + exp(-x));	
    } else {
	res = exp(x) / (1.0 + exp(x));
    }
    return res;
}


// Evaluate the gradient of the log partial likelihood wrt b
// [[Rcpp::export]]
arma::rowvec log_PL_grad(const arma::mat &X, const arma::vec &b, 
	const arma::uvec &Y_sorted, const arma::uvec &Y_failure)
{
    arma::vec xb = X * b;
    arma::rowvec res = arma::rowvec(b.n_rows, arma::fill::zeros);

    arma::rowvec num = arma::rowvec(b.n_rows, arma::fill::zeros);
    double den = 0.0;

    int ind_last_failure = X.n_rows;
    int ind_curr_failure = 0;

    for (int i = (Y_failure.n_rows - 1); i >= 0; --i) {
	ind_curr_failure = Y_failure(i);
	auto R = arma::span(ind_curr_failure, ind_last_failure - 1);

	den += sum(exp(xb(Y_sorted(R))));
	num += (exp(xb(Y_sorted(R))).t() * X.rows(Y_sorted(R)));
	res += X.row((Y_sorted(ind_curr_failure))) - num/den;

	ind_last_failure = ind_curr_failure;
    }

    return res;
}


// Evaluate the gradient of the log Laplace density wrt. b
arma::rowvec log_Laplace_grad(arma::vec beta, double lambda)
{
    return - lambda * sign(beta.t());
}


void leapfrog(arma::vec &b, arma::vec &r, double e, const arma::mat &X,
	const arma::uvec &Y_sorted, const arma::uvec &Y_failure, double lambda)
{
    r += 0.5 * e * log_PL_grad(X, b, Y_sorted, Y_failure).t() +
	 0.5 * e * log_Laplace_grad(b, lambda).t();
    b += e * r;
    r += 0.5 * e * log_PL_grad(X, b, Y_sorted, Y_failure).t() +
	 0.5 * e * log_Laplace_grad(b, lambda).t();
}
