% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmf1k1k_supervised_classification_variational_test(Kx, Kz, state)
    prediction.Gx.mean = state.Ax.mean' * Kx;

    prediction.Gz.mean = state.Az.mean' * Kz;

    prediction.F.mean = prediction.Gx.mean' * prediction.Gz.mean;
end