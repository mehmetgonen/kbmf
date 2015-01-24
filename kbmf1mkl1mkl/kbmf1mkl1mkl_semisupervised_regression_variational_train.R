# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1mkl1mkl_semisupervised_regression_variational_train <- function(Kx, Kz, Y, parameters) {
  set.seed(parameters$seed)

  Dx <- dim(Kx)[1]
  Nx <- dim(Kx)[2]
  Px <- dim(Kx)[3]
  Dz <- dim(Kz)[1]
  Nz <- dim(Kz)[2]
  Pz <- dim(Kz)[3]
  R <- parameters$R
  sigmag <- parameters$sigmag
  sigmah <- parameters$sigmah
  sigmay <- parameters$sigmay

  Lambdax <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dx, R), scale = matrix(parameters$beta_lambda, Dx, R))
  Ax <- list(mean = matrix(rnorm(Dx * R), Dx, R), covariance = array(diag(1, Dx, Dx), c(Dx, Dx, R)))
  Gx <- list(mean = array(rnorm(R * Nx * Px), c(R, Nx, Px)), covariance = array(diag(1, R, R), c(R, R, Px)))
  etax <- list(shape = matrix(parameters$alpha_eta + 0.5, Px, 1), scale = matrix(parameters$beta_eta, Px, 1))
  ex <- list(mean = matrix(1, Px, 1), covariance = diag(1, Px, Px))
  Hx <- list(mean = matrix(rnorm(R * Nx), R, Nx), covariance = array(diag(1, R, R), c(R, R, Nx)))

  Lambdaz <- list(shape = matrix(parameters$alpha_lambda + 0.5, Dz, R), scale = matrix(parameters$beta_lambda, Dz, R))
  Az <- list(mean = matrix(rnorm(Dz * R), Dz, R), covariance = array(diag(1, Dz, Dz), c(Dz, Dz, R)))
  Gz <- list(mean = array(rnorm(R * Nz * Pz), c(R, Nz, Pz)), covariance = array(diag(1, R, R), c(R, R, Pz)))
  etaz <- list(shape = matrix(parameters$alpha_eta + 0.5, Pz, 1), scale = matrix(parameters$beta_eta, Pz, 1))
  ez <- list(mean = matrix(1, Pz, 1), covariance = diag(1, Pz, Pz))
  Hz <- list(mean = matrix(rnorm(R * Nz), R, Nz), covariance = array(diag(1, R, R), c(R, R, Nz)))

  KxKx <- matrix(0, Dx, Dx)
  for (m in 1:Px) {
    KxKx <- KxKx + tcrossprod(Kx[,,m], Kx[,,m])
  }
  Kx <- matrix(Kx, Dx, Nx * Px)

  KzKz <- matrix(0, Dz, Dz)
  for (n in 1:Pz) {
    KzKz <- KzKz + tcrossprod(Kz[,,n], Kz[,,n])
  }
  Kz <- matrix(Kz, Dz, Nz * Pz)

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
      indices <- which(is.na(Y[i,]) == FALSE)
      Hx$covariance[,,i] <- chol2inv(chol(diag(1 / sigmah^2, R, R) + (tcrossprod(Hz$mean[,indices, drop = FALSE], Hz$mean[,indices, drop = FALSE]) + apply(Hz$covariance[,,indices, drop = FALSE], 1:2, sum)) / sigmay^2))
      Hx$mean[,i] <- tcrossprod(Hz$mean[,indices, drop = FALSE], Y[i, indices, drop = FALSE]) / sigmay^2
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
      Hz$covariance[,,j] <- chol2inv(chol(diag(1 / sigmah^2, R, R) + (tcrossprod(Hx$mean[,indices, drop = FALSE], Hx$mean[,indices, drop = FALSE]) + apply(Hx$covariance[,,indices, drop = FALSE], 1:2, sum)) / sigmay^2))
      Hz$mean[,j] <- Hx$mean[,indices, drop = FALSE] %*% Y[indices, j, drop = FALSE] / sigmay^2
      for (n in 1:Pz) {
        Hz$mean[,j] <- Hz$mean[,j] + ez$mean[n] * Gz$mean[,j,n] / sigmah^2
      }
      Hz$mean[,j] <- Hz$covariance[,,j] %*% Hz$mean[,j]
    }
  }

  state <- list(Lambdax = Lambdax, Ax = Ax, etax = etax, ex = ex, Lambdaz = Lambdaz, Az = Az, etaz = etaz, ez = ez, parameters = parameters)
}