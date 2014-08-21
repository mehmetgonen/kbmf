# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1k1mkl_supervised_regression_variational_test <- function(Kx, Kz, state) {
  Nz <- dim(Kz)[2]
  Pz <- dim(Kz)[3]
  R <- dim(state$Ax$mean)[2]

  Gx <- list(mean = crossprod(state$Ax$mean, Kx))

  Gz <- list(mean = array(0, c(R, Nz, Pz)))
  for (n in 1:Pz) {
    Gz$mean[,,n] <- crossprod(state$Az$mean, Kz[,,n])
  }
  Hz <- list(mean = matrix(0, R, Nz))
  for (n in 1:Pz) {
    Hz$mean <- Hz$mean + state$ez$mean[n] * Gz$mean[,,n]
  }

  Y <- list(mean = crossprod(Gx$mean, Hz$mean))

  prediction <- list(Gx = Gx, Gz = Gz, Hz = Hz, Y = Y)
}