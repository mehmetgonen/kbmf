# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1mkl1k_semisupervised_classification_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Px <- dim(Kx)[3]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  R <- parameters$R
  sigmag <- parameters$sigmag
  sigmah <- parameters$sigmah

  Lambdax <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dx, R), scale = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mean = matrix(rnorm(Dx * R), Dx, R), covariance = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mean = array(rnorm(R * Nx * Px), c(R, Nx, Px)), covariance = array(diag(1, R, R), c(R, R, Px)))
  etax <- list(shape = matrix(parameters$alpha_eta + 0.5, Px, 1), scale = matrix(parameters$beta_eta, Px, 1))
  ex <- list(mean = matrix(1, Px, 1), covariance = diag(1, Px, Px))
  Hx <- list(mean = matrix(rnorm(R * Nx), R, Nx), covariance = array(diag(1, R, R), c(R, R, Nx)))

  Lambdaz <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dz, R), scale = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mean = matrix(rnorm(Dz * R), Dz, R), covariance = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mean = matrix(rnorm(R * Nz), R, Nz), covariance = array(diag(1, R, R), c(R, R, Nz)))

  F <- list(mean = (abs(matrix(rnorm(Nx * Nz), Nx, Nz)) + parameters$margin) * sign(Y), covariance = matrix(1, Nx, Nz))

  KxKx <- matrix(0, Dx, Dx)
  for (m in 1:Px) {
    KxKx <- KxKx + tcrossprod(Kx[,,m], Kx[,,m])
  }
  Kx <- matrix(Kx, Dx, Nx * Px)

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
      Ax$mean[,s] <- Ax$covariance[,,s] %*% (Kx %*% matrix(Gx$mean[s,,], Nx * Px, 1) / sigmag^2)
    }
    # update Gx
    for (m in 1:Px) {
      Gx$covariance[,,m] <- chol2inv(chol(diag(1 / sigmag^2, R, R) + diag((ex$mean[m] * ex$mean[m] + ex$covariance[m, m]) / sigmah^2, R, R)))
      Gx$mean[,,m] <- crossprod(Ax$mean, Kx[,((m - 1) * Nx + 1):(m * Nx)]) / sigmag^2 + ex$mean[m] * Hx$mean / sigmah^2
      for (o in setdiff(1:Px, m)) {
        Gx$mean[,,m] <- Gx$mean[,,m] - (ex$mean[m] * ex$mean[o] + ex$covariance[m, o]) * Gx$mean[,,o] / sigmah^2
      }
      Gx$mean[,,m] <- Gx$covariance[,,m] %*% Gx$mean[,,m]
    }
    # update etax
    etax$scale <- 1 / (1 / parameters$beta_eta + 0.5 * (ex$mean^2 + diag(ex$covariance)))
    # update ex
    ex$covariance <- diag(as.vector(etax$shape * etax$scale))
    for (m in 1:Px) {
      for (o in 1:Px) {
        ex$covariance[m, o] <- ex$covariance[m, o] + (sum(Gx$mean[,,m] * Gx$mean[,,o]) + (m == o) * Nx * sum(diag(Gx$covariance[,,m]))) / sigmah^2
      }
    }
    ex$covariance <- chol2inv(chol(ex$covariance))
    for (m in 1:Px) {
      ex$mean[m] <- sum(Gx$mean[,,m] * Hx$mean) / sigmah^2
    }
    ex$mean <- ex$covariance %*% ex$mean
    # update Hx
    for (i in 1:Nx) {
      indices <- which(is.nan(Y[i,]) == FALSE)
      Hx$covariance[,,i] <- chol2inv(chol(diag(1 / sigmah^2, R, R) + tcrossprod(Gz$mean[,indices], Gz$mean[,indices]) + apply(Gz$covariance[,,indices], 1:2, sum)))
      Hx$mean[,i] <- tcrossprod(Gz$mean[,indices], F$mean[i, indices, drop = FALSE])
      for (m in 1:Px) {
        Hx$mean[,i] <- Hx$mean[,i] + ex$mean[m] * Gx$mean[,i,m] / sigmah^2
      }
      Hx$mean[,i] <- Hx$covariance[,,i] %*% Hx$mean[,i]
    }

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
    for (j in 1:Nz) {
      indices <- which(is.nan(Y[,j]) == FALSE)
      Gz$covariance[,,j] <- chol2inv(chol(diag(1 / sigmag^2, R, R) + tcrossprod(Hx$mean[,indices], Hx$mean[,indices]) + apply(Hx$covariance[,,indices], 1:2, sum)))
      Gz$mean[,j] <- Gz$covariance[,,j] %*% (crossprod(Az$mean, Kz[,j]) / sigmag^2 + Hx$mean[,indices] %*% F$mean[indices, j])
    }

    # update F
    output <- crossprod(Hx$mean, Gz$mean)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mean <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$covariance <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, etax = etax, ex = ex, Lambdaz = Lambdaz, Az = Az, parameters = parameters)
}