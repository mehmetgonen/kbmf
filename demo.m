rand('state', 1606); %#ok<RAND>
randn('state', 1606); %#ok<RAND>

Px = 15;
Nx = 40;
Pz = 10;
Nz = 60;

X = randn(Px, Nx);
Z = randn(Pz, Nz);
Y = X(1, :)' * Z(3, :) + X(4, :)' * Z(8, :) + X(7, :)' * Z(10, :) + 1 * randn(Nx, Nz);

Kx = zeros(Nx, Nx, Px);
for m = 1:Px
    Kx(:, :, m) = X(m, :)' * X(m, :);
end

Kz = zeros(Nz, Nz, Pz);
for n = 1:Pz
    Kz(:, :, n) = Z(n, :)' * Z(n, :);
end

state = kbmf_regression_train(Kx, Kz, Y, 5);
prediction = kbmf_regression_test(Kx, Kz, state);

fprintf(1, 'RMSE = %.4f\n', sqrt(mean(mean((prediction.Y.mu - Y).^2))));

figure;
set(gca, 'FontSize', 24);
imagesc(Y, [-9, +9]); colorbar; set(gca, 'FontSize', 24);
set(gca, 'XTick', []); set(gca, 'YTick', []); title('target outputs');

figure;
set(gca, 'FontSize', 24);
imagesc(prediction.Y.mu, [-9, +9]); colorbar; set(gca, 'FontSize', 24);
set(gca, 'XTick', []); set(gca, 'YTick', []); title('predicted outputs');

figure;
subplot(2, 1, 1); hold on; box on;
set(gca, 'FontSize', 24);
title('kernel weights on X');
xlim([0.5, 0.5 + Px]);
ylim([-0.25, 0.75]);
set(gca, 'XTick', 1:1:Px);
set(gca, 'XTickLabel', 1:1:Px);
set(gca, 'YTick', -0.25:0.25:0.75);
set(gca, 'YTickLabel', {'-0.25', '0.00', '0.25', '0.50', '0.75'});
bar(state.ex.mu);
subplot(2, 1, 2); hold on; box on;
set(gca, 'FontSize', 24);
title('kernel weights on Z');
xlim([0.5, 0.5 + Pz]);
ylim([-0.25, 0.75]);
set(gca, 'XTick', 1:1:Pz);
set(gca, 'XTickLabel', 1:1:Pz);
set(gca, 'YTick', -0.25:0.25:0.75);
set(gca, 'YTickLabel', {'-0.25', '0.00', '0.25', '0.50', '0.75'});
bar(state.ez.mu);
