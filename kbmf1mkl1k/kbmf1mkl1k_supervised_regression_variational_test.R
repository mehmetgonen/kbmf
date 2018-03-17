kbmf1mkl1k_supervised_regression_variational_test <- function(Kx, Kz, state) {
  Nx <- dim(Kx)[2]
  Px <- dim(Kx)[3]
  Nz <- dim(Kz)[2]
  R <- dim(state$Ax$mu)[2]

  Gx <- list(mu = array(0, c(R, Nx, Px)))
  for (m in 1:Px) {
    Gx$mu[,,m] <- crossprod(state$Ax$mu, Kx[,,m])
  }
  Hx <- list(mu = matrix(0, R, Nx))
  for (m in 1:Px) {
    Hx$mu <- Hx$mu + state$ex$mu[m] * Gx$mu[,,m]
  }

  Gz <- list(mu = crossprod(state$Az$mu, Kz))

  Y <- list(mu = crossprod(Hx$mu, Gz$mu))

  prediction <- list(Gx = Gx, Hx = Hx, Gz = Gz, Y = Y)
}
