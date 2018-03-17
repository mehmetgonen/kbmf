kbmf_classification_test <- function(Kx, Kz, state) {
  prediction <- state$parameters$test_function(Kx, Kz, state)
}
