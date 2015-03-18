% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmf1k1k_supervised_classification_variational_test(Kx, Kz, state)
    prediction.Gx.mu = state.Ax.mu' * Kx;

    prediction.Gz.mu = state.Az.mu' * Kz;

    prediction.F.mu = prediction.Gx.mu' * prediction.Gz.mu;
end