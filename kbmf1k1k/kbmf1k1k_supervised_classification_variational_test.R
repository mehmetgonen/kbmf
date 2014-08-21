# Mehmet Gonen (mehmet.gonen@gmail.com)

kbmf1k1k_supervised_classification_variational_test <- function(Kx, Kz, state) {
  Gx <- list(mean = crossprod(state$Ax$mean, Kx))

  Gz <- list(mean = crossprod(state$Az$mean, Kz))

  F <- list(mean = crossprod(Gx$mean, Gz$mean))

  prediction <- list(Gx = Gx, Gz = Gz, F = F)
}