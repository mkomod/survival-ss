Rcpp::sourceCpp("../src/sampler.cpp")

set.seed(1)
n <- 100; omega <- 1; censoring_lvl <- 0.4
b <- c(2, 2, rep(0, 248)); p <- length(b)
X <- matrix(rnorm(n * p), nrow=n)
y <- runif(nrow(X))
Y <- log(1 - y) / - (exp(X %*% b) * omega)
delta  <- runif(n) > censoring_lvl      # 0: censored, 1: uncensored
Y[!delta] <- Y[!delta] * runif(sum(!delta))

Y_sorted <- order(Y) - 1
Y_failure <- which(delta[Y_sorted + 1] == 1) - 1

lambda <- 0.1
kernel_sd <- .33
kernel_scale <- 10
mcmc_samples <- 5000
verbose <- T
fit <- sampler(Y_sorted, Y_failure, X, 0.3, kernel_sd, 
	       kernel_scale, mcmc_samples, verbose)

apply(fit$z, 1, mean)
apply(fit$b, 1, mean) |> round(2)

plot(fit$b[1, ], type="l")
plot(fit$b[2, ], type="l")
plot(fit$b[3, ], type="l")

acf(fit$b[1, ])
acf(fit$b[2, ])
acf(fit$b[3, ])

