source("kbmf_regression_train.R")
source("kbmf_regression_test.R")

set.seed(1606)

Px <- 15
Nx <- 40
Pz <- 10
Nz <- 60

X <- matrix(rnorm(Px * Nx), Px, Nx)
Z <- matrix(rnorm(Pz * Nz), Pz, Nz)
Y <- crossprod(X[1,,drop = FALSE], Z[3,,drop = FALSE]) + crossprod(X[4,,drop = FALSE], Z[8,,drop = FALSE]) + crossprod(X[7,,drop = FALSE], Z[10,,drop = FALSE]) + 1 * matrix(rnorm(Nx * Nz), Nx, Nz)

Kx <- array(0, c(Nx, Nx, Px))
for(m in 1:Px) {
  Kx[,, m] <- crossprod(X[m,,drop = FALSE], X[m,,drop = FALSE])
}

Kz <- array(0, c(Nz, Nz, Pz))
for(n in 1:Pz) {
  Kz[,, n] <- crossprod(Z[n,,drop = FALSE], Z[n,,drop = FALSE])
}

state <- kbmf_regression_train(Kx, Kz, Y, 5)
prediction <- kbmf_regression_test(Kx, Kz, state)

print(sprintf("RMSE = %.4f", sqrt(mean((prediction$Y$mu - Y)^2))))

print("kernel weights on X")
print(state$ex$mu)

print("kernel weights on Z")
print(state$ez$mu)
