
#' Run MCMC sampler
#'
#' @param Y observed failure / censoring times
#' @param delta censoring vector (0: censored, 1: uncensored)
#' @param X matrix of observations
#' @param lambda penalisation parameter
#' @param kernel_sd standard deviation of sampling kernel
#' @param mcmc_samples number of samples
#' @param verbose print a progress bar
#'
#' @examples
#' set.seed(1)
#' n <- 100; omega <- 1; censoring_lvl <- 0.4
#' b <- c(2, 2, rep(0, 148)); p <- length(b)
#' X <- matrix(rnorm(n * p), nrow=n)
#' y <- runif(nrow(X))
#' Y <- log(1 - y) / - (exp(X %*% b) * omega)
#' delta  <- runif(n) > censoring_lvl      # 0: censored, 1: uncensored
#' Y[!delta] <- Y[!delta] * runif(sum(!delta))
#' 
#' run_sampler(Y, delta, X, 2)
#'
#' @export
run_sampler <- function(Y, delta, X, lambda, kernel_sd=0.2, 
	mcmc_samples=5e3, verbose=T)
{
    if (!is.matrix(X)) stop("X is not a matrix") 
    if (lambda < 0) stop("lambda must be greater than 0")
    if (kernel_sd < 0) stop("kernel_sd must be greater than 0")

    Y_sorted <- order(Y) - 1
    Y_failure <- which(delta[Y_sorted] == 1) - 1

    return(
	sampler(Y_sorted, Y_failure, X, lambda, kernel_sd, mcmc_samples, verbose)
    )
}

