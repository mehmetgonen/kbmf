# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1mkl1k_semisupervised_classification_variational_test <- function(Kx, Kz, state) {
  Nx <- dim(Kx)[2]
  Px <- dim(Kx)[3]
  Nz <- dim(Kz)[2]
  R <- dim(state$Ax$mean)[2]

  Gx <- list(mean = array(0, c(R, Nx, Px)))
  for (m in 1:Px) {
    Gx$mean[,,m] <- crossprod(state$Ax$mean, Kx[,,m])
  }
  Hx <- list(mean = matrix(0, R, Nx))
  for (m in 1:Px) {
    Hx$mean <- Hx$mean + state$ex$mean[m] * Gx$mean[,,m]
  }

  Gz <- list(mean = crossprod(state$Az$mean, Kz))

  F <- list(mean = crossprod(Hx$mean, Gz$mean))

  prediction <- list(Gx = Gx, Hx = Hx, Gz = Gz, F = F)
}