% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmf1mkl1mkl_semisupervised_regression_variational_test(Kx, Kz, state)
    Nx = size(Kx, 2);
    Px = size(Kx, 3);
    Nz = size(Kz, 2);
    Pz = size(Kz, 3);
    R = size(state.Ax.mean, 2);

    prediction.Gx.mean = zeros(R, Nx, Px);
    for m = 1:Px
        prediction.Gx.mean(:, :, m) = state.Ax.mean' * Kx(:, :, m);
    end
    prediction.Hx.mean = zeros(R, Nx);
    for m = 1:Px
        prediction.Hx.mean = prediction.Hx.mean + state.ex.mean(m) * prediction.Gx.mean(:, :, m);
    end

    prediction.Gz.mean = zeros(R, Nz, Pz);
    for n = 1:Pz
        prediction.Gz.mean(:, :, n) = state.Az.mean' * Kz(:, :, n);
    end
    prediction.Hz.mean = zeros(R, Nz);
    for n = 1:Pz
        prediction.Hz.mean = prediction.Hz.mean + state.ez.mean(n) * prediction.Gz.mean(:, :, n);
    end

    prediction.Y.mean = prediction.Hx.mean' * prediction.Hz.mean;
end