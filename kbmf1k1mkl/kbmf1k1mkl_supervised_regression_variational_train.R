kbmf1k1mkl_supervised_regression_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  Pz <- dim(Kz)[3]
  R <- parameters$R
  sigma_g <- parameters$sigma_g
  sigma_h <- parameters$sigma_h
  sigma_y <- parameters$sigma_y

  Lambdax <- list(alpha = matrix(parameters$alpha_lambda + 0.5, Dx, R), beta = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mu = matrix(rnorm(Dx * R), Dx, R), sigma = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mu = matrix(rnorm(R * Nx), R, Nx), sigma = diag(1, R, R))

  Lambdaz <- list(alpha = matrix(parameters$alpha_lambda + 0.5, Dz, R), beta = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mu = matrix(rnorm(Dz * R), Dz, R), sigma = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mu = array(rnorm(R * Nz * Pz), c(R, Nz, Pz)), sigma = array(diag(1, R, R), c(R, R, Pz)))
  etaz <- list(alpha = matrix(parameters$alpha_eta + 0.5, Pz, 1), beta = matrix(parameters$beta_eta, Pz, 1))
  ez <- list(mu = matrix(1, Pz, 1), sigma = diag(1, Pz, Pz))
  Hz <- list(mu = matrix(rnorm(R * Nz), R, Nz), sigma = diag(1, R, R))

  KxKx <- tcrossprod(Kx, Kx)

  KzKz <- matrix(0, Dz, Dz)
  for (n in 1:Pz) {
    KzKz <- KzKz + tcrossprod(Kz[,,n], Kz[,,n])
  }
  Kz <- matrix(Kz, Dz, Nz * Pz)

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
    Gx$sigma <- chol2inv(chol(diag(1 / sigma_g^2, R, R) + (tcrossprod(Hz$mu, Hz$mu) + Nz * Hz$sigma) / sigma_y^2))
    Gx$mu <- Gx$sigma %*% (crossprod(Ax$mu, Kx) / sigma_g^2 + tcrossprod(Hz$mu, Y) / sigma_y^2)

    # update Lambdaz
    for (s in 1:R) {
      Lambdaz$beta[,s] <- 1 / (1 / parameters$beta_lambda + 0.5 * (Az$mu[,s]^2 + diag(Az$sigma[,,s])))
    }
    # update Az
    for (s in 1:R) {
      Az$sigma[,,s] <- chol2inv(chol(diag(as.vector(Lambdaz$alpha[,s] * Lambdaz$beta[,s]), Dz, Dz) + KzKz / sigma_g^2))
      Az$mu[,s] <- Az$sigma[,,s] %*% (Kz %*% matrix(Gz$mu[s,,], Nz * Pz, 1) / sigma_g^2)
    }
    # update Gz
    for (n in 1:Pz) {
      Gz$sigma[,,n] <- chol2inv(chol(diag(1 / sigma_g^2, R, R) + diag((ez$mu[n] * ez$mu[n] + ez$sigma[n, n]) / sigma_h^2, R, R)))
      Gz$mu[,,n] <- crossprod(Az$mu, Kz[,((n - 1) * Nz + 1):(n * Nz)]) / sigma_g^2 + ez$mu[n] * Hz$mu / sigma_h^2
      for (p in setdiff(1:Pz, n)) {
        Gz$mu[,,n] <- Gz$mu[,,n] - (ez$mu[n] * ez$mu[p] + ez$sigma[n, p]) * Gz$mu[,,p] / sigma_h^2
      }
      Gz$mu[,,n] <- Gz$sigma[,,n] %*% Gz$mu[,,n]
    }
    # update etaz
    etaz$beta <- 1 / (1 / parameters$beta_eta + 0.5 * (ez$mu^2 + diag(ez$sigma)))
    # update ez
    ez$sigma <- diag(as.vector(etaz$alpha * etaz$beta))
    for (n in 1:Pz) {
      for (p in 1:Pz) {
        ez$sigma[n, p] <- ez$sigma[n, p] + (sum(Gz$mu[,,n] * Gz$mu[,,p]) + (n == p) * Nz * sum(diag(Gz$sigma[,,n]))) / sigma_h^2
      }
    }
    ez$sigma <- chol2inv(chol(ez$sigma))
    for (n in 1:Pz) {
      ez$mu[n] <- sum(Gz$mu[,,n] * Hz$mu) / sigma_h^2
    }
    ez$mu <- ez$sigma %*% ez$mu
    # update Hz
    Hz$sigma <- chol2inv(chol(diag(1 / sigma_h^2, R, R) + (tcrossprod(Gx$mu, Gx$mu) + Nx * Gx$sigma) / sigma_y^2))
    Hz$mu <- Gx$mu %*% Y / sigma_y^2
    for (n in 1:Pz) {
      Hz$mu <- Hz$mu + ez$mu[n] * Gz$mu[,,n] / sigma_h^2
    }
    Hz$mu <- Hz$sigma %*% Hz$mu
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, Lambdaz = Lambdaz, Az = Az, etaz = etaz, ez = ez, parameters = parameters)
}
