function state = kbmf1k1mkl_semisupervised_regression_variational_train(Kx, Kz, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    Dx = size(Kx, 1);
    Nx = size(Kx, 2);
    Dz = size(Kz, 1);
    Nz = size(Kz, 2);
    Pz = size(Kz, 3);
    R = parameters.R;
    sigma_g = parameters.sigma_g;
    sigma_h = parameters.sigma_h;
    sigma_y = parameters.sigma_y;

    Lambdax.alpha = (parameters.alpha_lambda + 0.5) * ones(Dx, R);
    Lambdax.beta = parameters.beta_lambda * ones(Dx, R);
    Ax.mu = randn(Dx, R);
    Ax.sigma = repmat(eye(Dx, Dx), [1, 1, R]);
    Gx.mu = randn(R, Nx);
    Gx.sigma = repmat(eye(R, R), [1, 1, Nx]);

    Lambdaz.alpha = (parameters.alpha_lambda + 0.5) * ones(Dz, R);
    Lambdaz.beta = parameters.beta_lambda * ones(Dz, R);
    Az.mu = randn(Dz, R);
    Az.sigma = repmat(eye(Dz, Dz), [1, 1, R]);
    Gz.mu = randn(R, Nz, Pz);
    Gz.sigma = repmat(eye(R, R), [1, 1, Pz]);
    etaz.alpha = (parameters.alpha_eta + 0.5) * ones(Pz, 1);
    etaz.beta = parameters.beta_eta * ones(Pz, 1);
    ez.mu = ones(Pz, 1);
    ez.sigma = eye(Pz, Pz);
    Hz.mu = randn(R, Nz);
    Hz.sigma = repmat(eye(R, R), [1, 1, Nz]);

    KxKx = Kx * Kx';

    KzKz = zeros(Dz, Dz);
    for n = 1:Pz
        KzKz = KzKz + Kz(:, :, n) * Kz(:, :, n)';
    end
    Kz = reshape(Kz, [Dz, Nz * Pz]);

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
        Lambdax.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Ax.mu.^2 + reshape(Ax.sigma(lambdax_indices), Dx, R)));
        %%%% update Ax
        for s = 1:R
            Ax.sigma(:, :, s) = (diag(Lambdax.alpha(:, s) .* Lambdax.beta(:, s)) + KxKx / sigma_g^2) \ eye(Dx, Dx);
            Ax.mu(:, s) = Ax.sigma(:, :, s) * (Kx * Gx.mu(s, :)' / sigma_g^2);
        end
        %%%% update Gx
        for i = 1:Nx
            indices = ~isnan(Y(i, :));
            Gx.sigma(:, :, i) = (eye(R, R) / sigma_g^2 + (Hz.mu(:, indices) * Hz.mu(:, indices)' + sum(Hz.sigma(:, :, indices), 3)) / sigma_y^2) \ eye(R, R);
            Gx.mu(:, i) = Gx.sigma(:, :, i) * (Ax.mu' * Kx(:, i) / sigma_g^2 + Hz.mu(:, indices) * Y(i, indices)' / sigma_y^2);
        end
        
        %%%% update Lambdaz
        Lambdaz.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (Az.mu.^2 + reshape(Az.sigma(lambdaz_indices), Dz, R)));
        %%%% update Az
        for s = 1:R
            Az.sigma(:, :, s) = (diag(Lambdaz.alpha(:, s) .* Lambdaz.beta(:, s)) + KzKz / sigma_g^2) \ eye(Dz, Dz);
            Az.mu(:, s) = Az.sigma(:, :, s) * (Kz * reshape(squeeze(Gz.mu(s, :, :)), Nz * Pz, 1) / sigma_g^2);
        end
        %%%% update Gz
        for n = 1:Pz
            Gz.sigma(:, :, n) = (eye(R, R) / sigma_g^2 + (ez.mu(n) * ez.mu(n) + ez.sigma(n, n)) * eye(R, R) / sigma_h^2) \ eye(R, R);
            Gz.mu(:, :, n) = Az.mu' * Kz(:, (n - 1) * Nz + 1:n * Nz) / sigma_g^2 + ez.mu(n) * Hz.mu / sigma_h^2;
            for p = [1:n - 1, n + 1:Pz]
                Gz.mu(:, :, n) = Gz.mu(:, :, n) - (ez.mu(n) * ez.mu(p) + ez.sigma(n, p)) * Gz.mu(:, :, p) / sigma_h^2;
            end
            Gz.mu(:, :, n) = Gz.sigma(:, :, n) * Gz.mu(:, :, n);
        end
        %%%% update etaz
        etaz.beta = 1 ./ (1 / parameters.beta_eta + 0.5 * (ez.mu.^2 + diag(ez.sigma)));
        %%%% update ez
        ez.sigma = diag(etaz.alpha .* etaz.beta);
        for n = 1:Pz
            for p = 1:Pz
                ez.sigma(n, p) = ez.sigma(n, p) + (sum(sum(Gz.mu(:, :, n) .* Gz.mu(:, :, p))) + (n == p) * Nz * sum(diag(Gz.sigma(:, :, n)))) / sigma_h^2;
            end
        end
        ez.sigma = ez.sigma \ eye(Pz, Pz);
        for n = 1:Pz
            ez.mu(n) = sum(sum(Gz.mu(:, :, n) .* Hz.mu)) / sigma_h^2;
        end
        ez.mu = ez.sigma * ez.mu;
        %%%% update Hz
        for j = 1:Nz
            indices = ~isnan(Y(:, j));
            Hz.sigma(:, :, j) = (eye(R, R) / sigma_h^2 + (Gx.mu(:, indices) * Gx.mu(:, indices)' + sum(Gx.sigma(:, :, indices), 3)) / sigma_y^2) \ eye(R, R);
            Hz.mu(:, j) = Gx.mu(:, indices) * Y(indices, j) / sigma_y^2;
            for n = 1:Pz
                Hz.mu(:, j) = Hz.mu(:, j) + ez.mu(n) * Gz.mu(:, j, n) / sigma_h^2;
            end
            Hz.mu(:, j) = Hz.sigma(:, :, j) * Hz.mu(:, j);
        end
    end

    state.Lambdax = Lambdax;
    state.Ax = Ax;
    state.Lambdaz = Lambdaz;
    state.Az = Az;
    state.etaz = etaz;
    state.ez = ez;
    state.parameters = parameters;
end
