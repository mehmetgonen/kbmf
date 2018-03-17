kbmf1k1k_semisupervised_regression_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  R <- parameters$R
  sigma_g <- parameters$sigma_g
  sigma_y <- parameters$sigma_y

  Lambdax <- list(alpha = matrix(parameters$alpha_lambda + 0.5, Dx, R), beta = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mu = matrix(rnorm(Dx * R), Dx, R), sigma = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mu = matrix(rnorm(R * Nx), R, Nx), sigma = array(diag(1, R, R), c(R, R, Nx)))

  Lambdaz <- list(alpha = matrix(parameters$alpha_lambda + 0.5, Dz, R), beta = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mu = matrix(rnorm(Dz * R), Dz, R), sigma = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mu = matrix(rnorm(R * Nz), R, Nz), sigma = array(diag(1, R, R), c(R, R, Nz)))

  KxKx <- tcrossprod(Kx, Kx)
  KzKz <- tcrossprod(Kz, Kz)

  for (iter in 1:parameters$iteration) {
    # update Lambdax
    for (s in 1:R) {
      Lambdax$beta[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Ax$mu[,s]^2 + diag(Ax$sigma[,,s])))
    }
    # update Ax
    for (s in 1:R) {
      Ax$sigma[,,s] <- chol2inv(chol(diag(as.vector(Lambdax$alpha[,s] * Lambdax$beta[,s]), Dx, Dx) + KxKx / sigma_g^2))
      Ax$mu[,s] <- Ax$sigma[,,s] %*% (tcrossprod(Kx, Gx$mu[s,,drop = FALSE]) / sigma_g^2)
    }
    # update Gx
    for (i in 1:Nx) {
      indices <- which(is.na(Y[i,]) == FALSE)
      Gx$sigma[,,i] <- chol2inv(chol(diag(1 / sigma_g^2, R, R) + (tcrossprod(Gz$mu[,indices, drop = FALSE], Gz$mu[,indices, drop = FALSE]) + apply(Gz$sigma[,,indices, drop = FALSE], 1:2, sum)) / sigma_y^2))
      Gx$mu[,i] <- Gx$sigma[,,i] %*% (crossprod(Ax$mu, Kx[,i]) / sigma_g^2 + tcrossprod(Gz$mu[,indices, drop = FALSE], Y[i, indices, drop = FALSE]) / sigma_y^2)
    }

    # update Lambdaz
    for (s in 1:R) {
      Lambdaz$beta[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Az$mu[,s]^2 + diag(Az$sigma[,,s])))
    }
    # update Az
    for (s in 1:R) {
      Az$sigma[,,s] <- chol2inv(chol(diag(as.vector(Lambdaz$alpha[,s] * Lambdaz$beta[,s]), Dz, Dz) + KzKz / sigma_g^2))
      Az$mu[,s] <- Az$sigma[,,s] %*% (tcrossprod(Kz, Gz$mu[s,,drop = FALSE]) / sigma_g^2)
    }
    # update Gz
    for (j in 1:Nz) {
      indices <- which(is.na(Y[,j]) == FALSE)
      Gz$sigma[,,j] <- chol2inv(chol(diag(1 / sigma_g^2, R, R) + (tcrossprod(Gx$mu[,indices, drop = FALSE], Gx$mu[,indices, drop = FALSE]) + apply(Gx$sigma[,,indices, drop = FALSE], 1:2, sum)) / sigma_y^2))
      Gz$mu[,j] <- Gz$sigma[,,j] %*% (crossprod(Az$mu, Kz[,j]) / sigma_g^2 + Gx$mu[,indices, drop = FALSE] %*% Y[indices, j, drop = FALSE] / sigma_y^2)
    }
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, Lambdaz = Lambdaz, Az = Az, parameters = parameters)
}
