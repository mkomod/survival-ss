# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

sampler <- function(Y_sorted, Y_failure, X, lambda, a_0, b_0, kernel_sd, kernel_scale, mcmc_samples, verbose) {
    .Call(`_survival_ss_sampler`, Y_sorted, Y_failure, X, lambda, a_0, b_0, kernel_sd, kernel_scale, mcmc_samples, verbose)
}

log_PL <- function(X, b, Y_sorted, Y_failure) {
    .Call(`_survival_ss_log_PL`, X, b, Y_sorted, Y_failure)
}

log_PL_grad <- function(X, b, Y_sorted, Y_failure) {
    .Call(`_survival_ss_log_PL_grad`, X, b, Y_sorted, Y_failure)
}

