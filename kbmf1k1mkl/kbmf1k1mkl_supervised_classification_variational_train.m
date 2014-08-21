% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbmf1k1mkl_supervised_classification_variational_train(Kx, Kz, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    Dx = size(Kx, 1);
    Nx = size(Kx, 2);
    Dz = size(Kz, 1);
    Nz = size(Kz, 2);
    Pz = size(Kz, 3);
    R = parameters.R;
    sigmag = parameters.sigmag;
    sigmah = parameters.sigmah;

    Lambdax.shape = (parameters.alpha_lambda + 0.5) * ones(Dx, R);
    Lambdax.scale = parameters.beta_lambda * ones(Dx, R);
    Ax.mean = randn(Dx, R);
    Ax.covariance = repmat(eye(Dx, Dx), [1, 1, R]);
    Gx.mean = randn(R, Nx);
    Gx.covariance = eye(R, R);
    
    Lambdaz.shape = (parameters.alpha_lambda + 0.5) * ones(Dz, R);
    Lambdaz.scale = parameters.beta_lambda * ones(Dz, R);
    Az.mean = randn(Dz, R);
    Az.covariance = repmat(eye(Dz, Dz), [1, 1, R]);
    Gz.mean = randn(R, Nz, Pz);
    Gz.covariance = repmat(eye(R, R), [1, 1, Pz]);
    etaz.shape = (parameters.alpha_eta + 0.5) * ones(Pz, 1);
    etaz.scale = parameters.beta_eta * ones(Pz, 1);
    ez.mean = ones(Pz, 1);
    ez.covariance = eye(Pz, Pz);
    Hz.mean = randn(R, Nz);
    Hz.covariance = eye(R, R);

    F.mean = (abs(randn(Nx, Nz)) + parameters.margin) .* sign(Y);
    F.covariance = ones(Nx, Nz);

    KxKx = Kx * Kx';

    KzKz = zeros(Dz, Dz);
    for n = 1:Pz
        KzKz = KzKz + Kz(:, :, n) * Kz(:, :, n)';
    end
    Kz = reshape(Kz, [Dz, Nz * Pz]);

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
            Ax.mean(:, s) = Ax.covariance(:, :, s) * (Kx * Gx.mean(s, :)' / sigmag^2);
        end
        %%%% update Gx
        Gx.covariance = (eye(R, R) / sigmag^2 + Hz.mean * Hz.mean' + Nz * Hz.covariance) \ eye(R, R);
        Gx.mean = Gx.covariance * (Ax.mean' * Kx / sigmag^2 + Hz.mean * F.mean');

        %%%% update Lambdaz
        Lambdaz.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Az.mean.^2 + reshape(Az.covariance(lambdaz_indices), Dz, R)));
        %%%% update Az
        for s = 1:R
            Az.covariance(:, :, s) = (diag(Lambdaz.shape(:, s) .* Lambdaz.scale(:, s)) + KzKz / sigmag^2) \ eye(Dz, Dz);
            Az.mean(:, s) = Az.covariance(:, :, s) * (Kz * reshape(squeeze(Gz.mean(s, :, :)), Nz * Pz, 1) / sigmag^2);
        end
        %%%% update Gz
        for n = 1:Pz
            Gz.covariance(:, :, n) = (eye(R, R) / sigmag^2 + (ez.mean(n) * ez.mean(n) + ez.covariance(n, n)) * eye(R, R) / sigmah^2) \ eye(R, R);
            Gz.mean(:, :, n) = Az.mean' * Kz(:, (n - 1) * Nz + 1:n * Nz) / sigmag^2 + ez.mean(n) * Hz.mean / sigmah^2;
            for p = [1:n - 1, n + 1:Pz]
                Gz.mean(:, :, n) = Gz.mean(:, :, n) - (ez.mean(n) * ez.mean(p) + ez.covariance(n, p)) * Gz.mean(:, :, p) / sigmah^2;
            end
            Gz.mean(:, :, n) = Gz.covariance(:, :, n) * Gz.mean(:, :, n);
        end
        %%%% update etaz
        etaz.scale = 1 ./ (1 / parameters.beta_eta + 0.5 * (ez.mean.^2 + diag(ez.covariance)));
        %%%% update ez
        ez.covariance = diag(etaz.shape .* etaz.scale);
        for n = 1:Pz
            for p = 1:Pz
                ez.covariance(n, p) = ez.covariance(n, p) + (sum(sum(Gz.mean(:, :, n) .* Gz.mean(:, :, p))) + (n == p) * Nz * sum(diag(Gz.covariance(:, :, n)))) / sigmah^2;
            end
        end
        ez.covariance = ez.covariance \ eye(Pz, Pz);
        for n = 1:Pz
            ez.mean(n) = sum(sum(Gz.mean(:, :, n) .* Hz.mean)) / sigmah^2;
        end
        ez.mean = ez.covariance * ez.mean;
        %%%% update Hz
        Hz.covariance = (eye(R, R) / sigmah^2 + Gx.mean * Gx.mean' + Nx * Gx.covariance) \ eye(R, R);
        Hz.mean = Gx.mean * F.mean;
        for n = 1:Pz
            Hz.mean = Hz.mean + ez.mean(n) * Gz.mean(:, :, n) / sigmah^2;
        end
        Hz.mean = Hz.covariance * Hz.mean;

        %%%% update F
        output = Gx.mean' * Hz.mean;
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        F.mean = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        F.covariance = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;
    end

    state.Lambdax = Lambdax;
    state.Ax = Ax;
    state.Lambdaz = Lambdaz;
    state.Az = Az;
    state.etaz = etaz;
    state.ez = ez;
    state.parameters = parameters;
end