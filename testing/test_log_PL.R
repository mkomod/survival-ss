# Test the log-partial likelihood function
library(survival)
Rcpp::sourceCpp("../src/utils.cpp")

set.seed(1)
n <- 100; omega <- 1; censoring_lvl <- 0.4
b <- c(2, 2, rep(0, 18)); p <- length(b)
X <- matrix(rnorm(n * p), nrow=n)
y <- runif(nrow(X))
Y <- log(1 - y) / - (exp(X %*% b) * omega)
delta  <- runif(n) > censoring_lvl      # 0: censored, 1: uncensored
Y[!delta] <- Y[!delta] * runif(sum(!delta))

Y_sorted <- order(Y) - 1
Y_failure <- which(delta[Y_sorted + 1] == 1) - 1

tot <- 0
a <- max(X %*% b)
for (i in which(delta)) {
    R <- which(Y >= Y[i])    
    tot <- tot + X[i, ] %*% b - (a + log(sum(exp(X[R, ] %*% b - a))))
}
log_PL(X, coef(f), Y_sorted, Y_failure)


d <- rep(0, p)
for (i in which(delta)) {
    R <- which(Y >= Y[i])    
    d <- d + X[i, ] - t(exp(X[R,] %*% b)) %*% X[R, ] / sum(exp(X[R, ] %*% b))
    print(t(exp(X[R,] %*% b)) %*% X[R, ] / sum(exp(X[R, ] %*% b)))
}
d
log_PL_grad(X, coef(f), Y_sorted, Y_failure)


init <- rnorm(p)
o <- optim(init,
      fn=function(x) -as.numeric(log_PL(X, x, Y_sorted, Y_failure)),
      gr=function(x) -log_PL_grad(X, x, Y_sorted, Y_failure),
      method="BFGS")
o$par

f <- coxph(Surv(Y, delta) ~ X)
as.numeric(coef(f))

