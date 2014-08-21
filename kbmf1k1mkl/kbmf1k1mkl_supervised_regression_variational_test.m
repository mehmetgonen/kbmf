% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmf1k1mkl_supervised_regression_variational_test(Kx, Kz, state)
    Nz = size(Kz, 2);
    Pz = size(Kz, 3);
    R = size(state.Ax.mean, 2);

    prediction.Gx.mean = state.Ax.mean' * Kx;

    prediction.Gz.mean = zeros(R, Nz, Pz);
    for n = 1:Pz
        prediction.Gz.mean(:, :, n) = state.Az.mean' * Kz(:, :, n);
    end
    prediction.Hz.mean = zeros(R, Nz);
    for n = 1:Pz
        prediction.Hz.mean = prediction.Hz.mean + state.ez.mean(n) * prediction.Gz.mean(:, :, n);
    end

    prediction.Y.mean = prediction.Gx.mean' * prediction.Hz.mean;
end