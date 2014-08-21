# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1k1k_supervised_classification_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  R <- parameters$R
  sigmag <- parameters$sigmag

  Lambdax <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dx, R), scale = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mean = matrix(rnorm(Dx * R), Dx, R), covariance = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mean = matrix(rnorm(R * Nx), R, Nx), covariance = diag(1, R, R))

  Lambdaz <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dz, R), scale = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mean = matrix(rnorm(Dz * R), Dz, R), covariance = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mean = matrix(rnorm(R * Nz), R, Nz), covariance = diag(1, R, R))

  F <- list(mean = (abs(matrix(rnorm(Nx * Nz), Nx, Nz)) + parameters$margin) * sign(Y), covariance = matrix(1, Nx, Nz))

  KxKx <- tcrossprod(Kx, Kx)
  KzKz <- tcrossprod(Kz, Kz)

  lower <- matrix(-1e40, Nx, Nz)
  lower[which(Y > 0)] <- +parameters$margin
  upper <- matrix(+1e40, Nx, Nz)
  upper[which(Y < 0)] <- -parameters$margin

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
    Gx$covariance <- chol2inv(chol(diag(1 / sigmag^2, R, R) + tcrossprod(Gz$mean, Gz$mean) + Nz * Gz$covariance))
    Gx$mean <- Gx$covariance %*% (crossprod(Ax$mean, Kx) / sigmag^2 + tcrossprod(Gz$mean, F$mean))

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
    Gz$covariance <- chol2inv(chol(diag(1 / sigmag^2, R, R) + tcrossprod(Gx$mean, Gx$mean) + Nx * Gx$covariance))
    Gz$mean <- Gz$covariance %*% (crossprod(Az$mean, Kz) / sigmag^2 + Gx$mean %*% F$mean)

    # update F
    output <- crossprod(Gx$mean, Gz$mean)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mean <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$covariance <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, Lambdaz = Lambdaz, Az = Az, parameters = parameters)
}