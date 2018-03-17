function prediction = kbmf1mkl1k_semisupervised_classification_variational_test(Kx, Kz, state)
    Nx = size(Kx, 2);
    Px = size(Kx, 3);
    Nz = size(Kz, 2);
    R = size(state.Ax.mu, 2);

    prediction.Gx.mu = zeros(R, Nx, Px);
    for m = 1:Px
        prediction.Gx.mu(:, :, m) = state.Ax.mu' * Kx(:, :, m);
    end
    prediction.Hx.mu = zeros(R, Nx);
    for m = 1:Px
        prediction.Hx.mu = prediction.Hx.mu + state.ex.mu(m) * prediction.Gx.mu(:, :, m);
    end

    prediction.Gz.mu = zeros(R, Nz);
    for s = 1:R
        prediction.Gz.mu(s, :) = state.Az.mu(:, s)' * Kz;
    end

    prediction.F.mu = prediction.Hx.mu' * prediction.Gz.mu;
end
