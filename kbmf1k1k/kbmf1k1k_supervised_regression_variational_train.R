# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1k1k_supervised_regression_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  R <- parameters$R
  sigmag <- parameters$sigmag
  sigmay <- parameters$sigmay

  Lambdax <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dx, R), scale = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mean = matrix(rnorm(Dx * R), Dx, R), covariance = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mean = matrix(rnorm(R * Nx), R, Nx), covariance = diag(1, R, R))

  Lambdaz <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dz, R), scale = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mean = matrix(rnorm(Dz * R), Dz, R), covariance = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mean = matrix(rnorm(R * Nz), R, Nz), covariance = diag(1, R, R))

  KxKx <- tcrossprod(Kx, Kx)
  KzKz <- tcrossprod(Kz, Kz)

  for (iter in 1:parameters$iteration) {
    # update Lambdax
    for (s in 1:R) {
      Lambdax$scale[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Ax$mean[,s]^2 + diag(Ax$covariance[,,s])))
    }
    # update Ax
    for (s in 1:R) {
      Ax$covariance[,,s] <- chol2inv(chol(diag(as.vector(Lambdax$shape[,s] * Lambdax$scale[,s]), Dx, Dx) + KxKx / sigmag^2))
      Ax$mean[,s] <- Ax$covariance[,,s] %*% (tcrossprod(Kx, Gx$mean[s,,drop = FALSE]) / sigmag^2)
    }
    # update Gx
    Gx$covariance <- chol2inv(chol(diag(1 / sigmag^2, R, R) + (tcrossprod(Gz$mean, Gz$mean) + Nz * Gz$covariance) / sigmay^2))
    Gx$mean <- Gx$covariance %*% (crossprod(Ax$mean, Kx) / sigmag^2 + tcrossprod(Gz$mean, Y) / sigmay^2)

    # update Lambdaz
    for (s in 1:R) {
      Lambdaz$scale[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Az$mean[,s]^2 + diag(Az$covariance[,,s])))
    }
    # update Az
    for (s in 1:R) {
      Az$covariance[,,s] <- chol2inv(chol(diag(as.vector(Lambdaz$shape[,s] * Lambdaz$scale[,s]), Dz, Dz) + KzKz / sigmag^2))
      Az$mean[,s] <- Az$covariance[,,s] %*% (tcrossprod(Kz, Gz$mean[s,,drop = FALSE]) / sigmag^2)
    }
    # update Gz
    Gz$covariance <- chol2inv(chol(diag(1 / sigmag^2, R, R) + (tcrossprod(Gx$mean, Gx$mean) + Nx * Gx$covariance) / sigmay^2))
    Gz$mean <- Gz$covariance %*% (crossprod(Az$mean, Kz) / sigmag^2 + Gx$mean %*% Y / sigmay^2)
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, Lambdaz = Lambdaz, Az = Az, parameters = parameters)
}
