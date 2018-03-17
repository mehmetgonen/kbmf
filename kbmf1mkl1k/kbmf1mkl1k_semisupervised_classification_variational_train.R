kbmf1mkl1k_semisupervised_classification_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Px <- dim(Kx)[3]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  R <- parameters$R
  sigma_g <- parameters$sigma_g
  sigma_h <- parameters$sigma_h

  Lambdax <- list(alpha = matrix(parameters$alpha_lambda + 0.5, Dx, R), beta = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mu = matrix(rnorm(Dx * R), Dx, R), sigma = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mu = array(rnorm(R * Nx * Px), c(R, Nx, Px)), sigma = array(diag(1, R, R), c(R, R, Px)))
  etax <- list(alpha = matrix(parameters$alpha_eta + 0.5, Px, 1), beta = matrix(parameters$beta_eta, Px, 1))
  ex <- list(mu = matrix(1, Px, 1), sigma = diag(1, Px, Px))
  Hx <- list(mu = matrix(rnorm(R * Nx), R, Nx), sigma = array(diag(1, R, R), c(R, R, Nx)))

  Lambdaz <- list(alpha = matrix(parameters$alpha_lambda + 0.5, Dz, R), beta = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mu = matrix(rnorm(Dz * R), Dz, R), sigma = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mu = matrix(rnorm(R * Nz), R, Nz), sigma = array(diag(1, R, R), c(R, R, Nz)))

  F <- list(mu = (abs(matrix(rnorm(Nx * Nz), Nx, Nz)) + parameters$margin) * sign(Y), sigma = matrix(1, Nx, Nz))

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
      Lambdax$beta[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Ax$mu[,s]^2 + diag(Ax$sigma[,,s])))
    }
    # update Ax
    for (s in 1:R) {
      Ax$sigma[,,s] <- chol2inv(chol(diag(as.vector(Lambdax$alpha[,s] * Lambdax$beta[,s]), Dx, Dx) + KxKx / sigma_g^2))
      Ax$mu[,s] <- Ax$sigma[,,s] %*% (Kx %*% matrix(Gx$mu[s,,], Nx * Px, 1) / sigma_g^2)
    }
    # update Gx
    for (m in 1:Px) {
      Gx$sigma[,,m] <- chol2inv(chol(diag(1 / sigma_g^2, R, R) + diag((ex$mu[m] * ex$mu[m] + ex$sigma[m, m]) / sigma_h^2, R, R)))
      Gx$mu[,,m] <- crossprod(Ax$mu, Kx[,((m - 1) * Nx + 1):(m * Nx)]) / sigma_g^2 + ex$mu[m] * Hx$mu / sigma_h^2
      for (o in setdiff(1:Px, m)) {
        Gx$mu[,,m] <- Gx$mu[,,m] - (ex$mu[m] * ex$mu[o] + ex$sigma[m, o]) * Gx$mu[,,o] / sigma_h^2
      }
      Gx$mu[,,m] <- Gx$sigma[,,m] %*% Gx$mu[,,m]
    }
    # update etax
    etax$beta <- 1 / (1 / parameters$beta_eta + 0.5 * (ex$mu^2 + diag(ex$sigma)))
    # update ex
    ex$sigma <- diag(as.vector(etax$alpha * etax$beta))
    for (m in 1:Px) {
      for (o in 1:Px) {
        ex$sigma[m, o] <- ex$sigma[m, o] + (sum(Gx$mu[,,m] * Gx$mu[,,o]) + (m == o) * Nx * sum(diag(Gx$sigma[,,m]))) / sigma_h^2
      }
    }
    ex$sigma <- chol2inv(chol(ex$sigma))
    for (m in 1:Px) {
      ex$mu[m] <- sum(Gx$mu[,,m] * Hx$mu) / sigma_h^2
    }
    ex$mu <- ex$sigma %*% ex$mu
    # update Hx
    for (i in 1:Nx) {
      indices <- which(is.na(Y[i,]) == FALSE)
      Hx$sigma[,,i] <- chol2inv(chol(diag(1 / sigma_h^2, R, R) + tcrossprod(Gz$mu[,indices, drop = FALSE], Gz$mu[,indices, drop = FALSE]) + apply(Gz$sigma[,,indices, drop = FALSE], 1:2, sum)))
      Hx$mu[,i] <- tcrossprod(Gz$mu[,indices, drop = FALSE], F$mu[i, indices, drop = FALSE])
      for (m in 1:Px) {
        Hx$mu[,i] <- Hx$mu[,i] + ex$mu[m] * Gx$mu[,i,m] / sigma_h^2
      }
      Hx$mu[,i] <- Hx$sigma[,,i] %*% Hx$mu[,i]
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
      Gz$sigma[,,j] <- chol2inv(chol(diag(1 / sigma_g^2, R, R) + tcrossprod(Hx$mu[,indices, drop = FALSE], Hx$mu[,indices, drop = FALSE]) + apply(Hx$sigma[,,indices, drop = FALSE], 1:2, sum)))
      Gz$mu[,j] <- Gz$sigma[,,j] %*% (crossprod(Az$mu, Kz[,j]) / sigma_g^2 + Hx$mu[,indices, drop = FALSE] %*% F$mu[indices, j, drop = FALSE])
    }

    # update F
    output <- crossprod(Hx$mu, Gz$mu)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mu <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$sigma <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, etax = etax, ex = ex, Lambdaz = Lambdaz, Az = Az, parameters = parameters)
}
