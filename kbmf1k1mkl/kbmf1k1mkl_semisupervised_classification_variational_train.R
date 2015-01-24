# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1k1mkl_semisupervised_classification_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  Pz <- dim(Kz)[3]
  R <- parameters$R
  sigmag <- parameters$sigmag
  sigmah <- parameters$sigmah

  Lambdax <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dx, R), scale = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mean = matrix(rnorm(Dx * R), Dx, R), covariance = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mean = matrix(rnorm(R * Nx), R, Nx), covariance = array(diag(1, R, R), c(R, R, Nx)))

  Lambdaz <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dz, R), scale = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mean = matrix(rnorm(Dz * R), Dz, R), covariance = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mean = array(rnorm(R * Nz * Pz), c(R, Nz, Pz)), covariance = array(diag(1, R, R), c(R, R, Pz)))
  etaz <- list(shape = matrix(parameters$alpha_eta + 0.5, Pz, 1), scale = matrix(parameters$beta_eta, Pz, 1))
  ez <- list(mean = matrix(1, Pz, 1), covariance = diag(1, Pz, Pz))
  Hz <- list(mean = matrix(rnorm(R * Nz), R, Nz), covariance = array(diag(1, R, R), c(R, R, Nz)))

  F <- list(mean = (abs(matrix(rnorm(Nx * Nz), Nx, Nz)) + parameters$margin) * sign(Y), covariance = matrix(1, Nx, Nz))

  KxKx <- tcrossprod(Kx, Kx)

  KzKz <- matrix(0, Dz, Dz)
  for (n in 1:Pz) {
    KzKz <- KzKz + tcrossprod(Kz[,,n], Kz[,,n])
  }
  Kz <- matrix(Kz, Dz, Nz * Pz)

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
    for (i in 1:Nx) {
      indices <- which(is.na(Y[i,]) == FALSE)
      Gx$covariance[,,i] <- chol2inv(chol(diag(1 / sigmag^2, R, R) + tcrossprod(Hz$mean[,indices, drop = FALSE], Hz$mean[,indices, drop = FALSE]) + apply(Hz$covariance[,,indices, drop = FALSE], 1:2, sum)))
      Gx$mean[,i] <- Gx$covariance[,,i] %*% (crossprod(Ax$mean, Kx[,i]) / sigmag^2 + tcrossprod(Hz$mean[,indices, drop = FALSE], F$mean[i, indices, drop = FALSE]))
    }

    # update Lambdaz
    for (s in 1:R) {
      Lambdaz$scale[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Az$mean[,s]^2 + diag(Az$covariance[,,s])))
    }
    # update Az
    for (s in 1:R) {
      Az$covariance[,,s] <- chol2inv(chol(diag(as.vector(Lambdaz$shape[,s] * Lambdaz$scale[,s]), Dz, Dz) + KzKz / sigmag^2))
      Az$mean[,s] <- Az$covariance[,,s] %*% (Kz %*% matrix(Gz$mean[s,,], Nz * Pz, 1) / sigmag^2)
    }
    # update Gz
    for (n in 1:Pz) {
      Gz$covariance[,,n] <- chol2inv(chol(diag(1 / sigmag^2, R, R) + diag((ez$mean[n] * ez$mean[n] + ez$covariance[n, n]) / sigmah^2, R, R)))
      Gz$mean[,,n] <- crossprod(Az$mean, Kz[,((n - 1) * Nz + 1):(n * Nz)]) / sigmag^2 + ez$mean[n] * Hz$mean / sigmah^2
      for (p in setdiff(1:Pz, n)) {
        Gz$mean[,,n] <- Gz$mean[,,n] - (ez$mean[n] * ez$mean[p] + ez$covariance[n, p]) * Gz$mean[,,p] / sigmah^2
      }
      Gz$mean[,,n] <- Gz$covariance[,,n] %*% Gz$mean[,,n]
    }
    # update etaz
    etaz$scale <- 1 / (1 / parameters$beta_eta + 0.5 * (ez$mean^2 + diag(ez$covariance)))
    # update ez
    ez$covariance <- diag(as.vector(etaz$shape * etaz$scale))
    for (n in 1:Pz) {
      for (p in 1:Pz) {
        ez$covariance[n, p] <- ez$covariance[n, p] + (sum(Gz$mean[,,n] * Gz$mean[,,p]) + (n == p) * Nz * sum(diag(Gz$covariance[,,n]))) / sigmah^2
      }
    }
    ez$covariance <- chol2inv(chol(ez$covariance))
    for (n in 1:Pz) {
      ez$mean[n] <- sum(Gz$mean[,,n] * Hz$mean) / sigmah^2
    }
    ez$mean <- ez$covariance %*% ez$mean
    # update Hz
    for (j in 1:Nz) {
      indices <- which(is.na(Y[,j]) == FALSE)
      Hz$covariance[,,j] <- chol2inv(chol(diag(1 / sigmah^2, R, R) + tcrossprod(Gx$mean[,indices, drop = FALSE], Gx$mean[,indices, drop = FALSE]) + apply(Gx$covariance[,,indices, drop = FALSE], 1:2, sum)))
      Hz$mean[,j] <- Gx$mean[,indices, drop = FALSE] %*% F$mean[indices, j, drop = FALSE]
      for (n in 1:Pz) {
        Hz$mean[,j] <- Hz$mean[,j] + ez$mean[n] * Gz$mean[,j,n] / sigmah^2
      }
      Hz$mean[,j] <- Hz$covariance[,,j] %*% Hz$mean[,j]
    }

    # update F
    output <- crossprod(Gx$mean, Hz$mean)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mean <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$covariance <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, Lambdaz = Lambdaz, Az = Az, etaz = etaz, ez = ez, parameters = parameters)
}