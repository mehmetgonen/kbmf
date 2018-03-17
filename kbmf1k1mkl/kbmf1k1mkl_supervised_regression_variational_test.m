function prediction = kbmf1k1mkl_supervised_regression_variational_test(Kx, Kz, state)
    Nz = size(Kz, 2);
    Pz = size(Kz, 3);
    R = size(state.Ax.mu, 2);

    prediction.Gx.mu = state.Ax.mu' * Kx;

    prediction.Gz.mu = zeros(R, Nz, Pz);
    for n = 1:Pz
        prediction.Gz.mu(:, :, n) = state.Az.mu' * Kz(:, :, n);
    end
    prediction.Hz.mu = zeros(R, Nz);
    for n = 1:Pz
        prediction.Hz.mu = prediction.Hz.mu + state.ez.mu(n) * prediction.Gz.mu(:, :, n);
    end

    prediction.Y.mu = prediction.Gx.mu' * prediction.Hz.mu;
end
