% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmf1mkl1k_supervised_classification_variational_test(Kx, Kz, state)
    Nx = size(Kx, 2);
    Px = size(Kx, 3);
    Nz = size(Kz, 2);
    R = size(state.Ax.mean, 2);

    prediction.Gx.mean = zeros(R, Nx, Px);
    for m = 1:Px
        prediction.Gx.mean(:, :, m) = state.Ax.mean' * Kx(:, :, m);
    end
    prediction.Hx.mean = zeros(R, Nx);
    for m = 1:Px
        prediction.Hx.mean = prediction.Hx.mean + state.ex.mean(m) * prediction.Gx.mean(:, :, m);
    end

    prediction.Gz.mean = zeros(R, Nz);
    for s = 1:R
        prediction.Gz.mean(s, :) = state.Az.mean(:, s)' * Kz;
    end

    prediction.F.mean = prediction.Hx.mean' * prediction.Gz.mean;
end