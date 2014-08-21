% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbmf1mkl1k_semisupervised_classification_variational_train(Kx, Kz, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    Dx = size(Kx, 1);
    Nx = size(Kx, 2);
    Px = size(Kx, 3);
    Dz = size(Kz, 1);
    Nz = size(Kz, 2);
    R = parameters.R;
    sigmag = parameters.sigmag;
    sigmah = parameters.sigmah;

    Lambdax.shape = (parameters.alpha_lambda + 0.5) * ones(Dx, R);
    Lambdax.scale = parameters.beta_lambda * ones(Dx, R);
    Ax.mean = randn(Dx, R);
    Ax.covariance = repmat(eye(Dx, Dx), [1, 1, R]);
    Gx.mean = randn(R, Nx, Px);
    Gx.covariance = repmat(eye(R, R), [1, 1, Px]);
    etax.shape = (parameters.alpha_eta + 0.5) * ones(Px, 1);
    etax.scale = parameters.beta_eta * ones(Px, 1);
    ex.mean = ones(Px, 1);
    ex.covariance = eye(Px, Px);
    Hx.mean = randn(R, Nx);
    Hx.covariance = repmat(eye(R, R), [1, 1, Nx]);

    Lambdaz.shape = (parameters.alpha_lambda + 0.5) * ones(Dz, R);
    Lambdaz.scale = parameters.beta_lambda * ones(Dz, R);
    Az.mean = randn(Dz, R);
    Az.covariance = repmat(eye(Dz, Dz), [1, 1, R]);
    Gz.mean = randn(R, Nz);
    Gz.covariance = repmat(eye(R, R), [1, 1, Nz]);

    F.mean = (abs(randn(Nx, Nz)) + parameters.margin) .* sign(Y);
    F.covariance = ones(Nx, Nz);

    KxKx = zeros(Dx, Dx);
    for m = 1:Px
        KxKx = KxKx + Kx(:, :, m) * Kx(:, :, m)';
    end
    Kx = reshape(Kx, [Dx, Nx * Px]);

    KzKz = Kz * Kz';

    lower = -1e40 * ones(Nx, Nz);
    lower(Y > 0) = +parameters.margin;
    upper = +1e40 * ones(Nx, Nz);
    upper(Y < 0) = -parameters.margin;

    lambdax_indices = repmat(logical(eye(Dx, Dx)), [1, 1, R]);
    lambdaz_indices = repmat(logical(eye(Dz, Dz)), [1, 1, R]);

    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update Lambdax
        Lambdax.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Ax.mean.^2 + reshape(Ax.covariance(lambdax_indices), Dx, R)));
        %%%% update Ax
        for s = 1:R
            Ax.covariance(:, :, s) = (diag(Lambdax.shape(:, s) .* Lambdax.scale(:, s)) + KxKx / sigmag^2) \ eye(Dx, Dx);
            Ax.mean(:, s) = Ax.covariance(:, :, s) * (Kx * reshape(squeeze(Gx.mean(s, :, :)), Nx * Px, 1) / sigmag^2);
        end
        %%%% update Gx
        for m = 1:Px
            Gx.covariance(:, :, m) = (eye(R, R) / sigmag^2 + (ex.mean(m) * ex.mean(m) + ex.covariance(m, m)) * eye(R, R) / sigmah^2) \ eye(R, R);
            Gx.mean(:, :, m) = Ax.mean' * Kx(:, (m - 1) * Nx + 1:m * Nx) / sigmag^2 + ex.mean(m) * Hx.mean / sigmah^2;
            for o = [1:m - 1, m + 1:Px]
                Gx.mean(:, :, m) = Gx.mean(:, :, m) - (ex.mean(m) * ex.mean(o) + ex.covariance(m, o)) * Gx.mean(:, :, o) / sigmah^2;
            end
            Gx.mean(:, :, m) = Gx.covariance(:, :, m) * Gx.mean(:, :, m);
        end
        %%%% update etax
        etax.scale = 1 ./ (1 / parameters.beta_eta + 0.5 * (ex.mean.^2 + diag(ex.covariance)));
        %%%% update ex
        ex.covariance = diag(etax.shape .* etax.scale);
        for m = 1:Px
            for o = 1:Px
                ex.covariance(m, o) = ex.covariance(m, o) + (sum(sum(Gx.mean(:, :, m) .* Gx.mean(:, :, o))) + (m == o) * Nx * sum(diag(Gx.covariance(:, :, m)))) / sigmah^2;
            end
        end
        ex.covariance = ex.covariance \ eye(Px, Px);
        for m = 1:Px
            ex.mean(m) = sum(sum(Gx.mean(:, :, m) .* Hx.mean)) / sigmah^2;
        end
        ex.mean = ex.covariance * ex.mean;
        %%%% update Hx
        for i = 1:Nx
            indices = ~isnan(Y(i, :));
            Hx.covariance(:, :, i) = (eye(R, R) / sigmah^2 + Gz.mean(:, indices) * Gz.mean(:, indices)' + sum(Gz.covariance(:, :, indices), 3)) \ eye(R, R);
            Hx.mean(:, i) = Gz.mean(:, indices) * F.mean(i, indices)';
            for m = 1:Px
                Hx.mean(:, i) = Hx.mean(:, i) + ex.mean(m) * Gx.mean(:, i, m) / sigmah^2;
            end
            Hx.mean(:, i) = Hx.covariance(:, :, i) * Hx.mean(:, i);
        end

        %%%% update Lambdaz
        Lambdaz.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Az.mean.^2 + reshape(Az.covariance(lambdaz_indices), Dz, R)));
        %%%% update Az
        for s = 1:R
            Az.covariance(:, :, s) = (diag(Lambdaz.shape(:, s) .* Lambdaz.scale(:, s)) + KzKz / sigmag^2) \ eye(Dz, Dz);
            Az.mean(:, s) = Az.covariance(:, :, s) * (Kz * Gz.mean(s, :)' / sigmag^2);
        end        
        %%%% update Gz
        for j = 1:Nz
            indices = ~isnan(Y(:, j));
            Gz.covariance(:, :, j) = (eye(R, R) / sigmag^2 + Hx.mean(:, indices) * Hx.mean(:, indices)' + sum(Hx.covariance(:, :, indices), 3)) \ eye(R, R);
            Gz.mean(:, j) = Gz.covariance(:, :, j) * (Az.mean' * Kz(:, j) / sigmag^2 + Hx.mean(:, indices) * F.mean(indices, j));
        end

        %%%% update F
        output = Hx.mean' * Gz.mean;
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        F.mean = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        F.covariance = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
    end

    state.Lambdax = Lambdax;
    state.Ax = Ax;
    state.etax = etax;
    state.ex = ex;
    state.Lambdaz = Lambdaz;
    state.Az = Az;
    state.parameters = parameters;
end