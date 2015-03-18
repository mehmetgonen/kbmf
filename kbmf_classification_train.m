% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = kbmf_classification_train(Kx, Kz, Y, R, varargin)
    directory = fileparts(mfilename('fullpath'));
    addpath([directory, '/kbmf1k1k']);
    addpath([directory, '/kbmf1k1mkl']);
    addpath([directory, '/kbmf1mkl1k']);
    addpath([directory, '/kbmf1mkl1mkl']);
    Px = size(Kx, 3);
    Pz = size(Kz, 3);
    is_supervised = all(~isnan(Y(:)));

    parameters.alpha_lambda = 1;
    parameters.beta_lambda = 1;
    if Px > 1 || Pz > 1
        parameters.alpha_eta = 1;
        parameters.beta_eta = 1;
    end
    parameters.iteration = 200;
    parameters.margin = 1;
    parameters.progress = 1;
    parameters.R = R;
    parameters.seed = 1606;
    parameters.sigma_g = 0.1;
    if Px > 1 || Pz > 1
        parameters.sigma_h = 0.1;
    end

    if is_supervised == 1
        if Px == 1 && Pz == 1
            train_function = @kbmf1k1k_supervised_classification_variational_train;
            test_function = @kbmf1k1k_supervised_classification_variational_test;
        elseif Px > 1 && Pz == 1
            train_function = @kbmf1mkl1k_supervised_classification_variational_train;
            test_function = @kbmf1mkl1k_supervised_classification_variational_test;
        elseif Px == 1 && Pz > 1
            train_function = @kbmf1k1mkl_supervised_classification_variational_train;
            test_function = @kbmf1k1mkl_supervised_classification_variational_test;
        elseif Px > 1 && Pz > 1
            train_function = @kbmf1mkl1mkl_supervised_classification_variational_train;
            test_function = @kbmf1mkl1mkl_supervised_classification_variational_test;
        end
    else
        if Px == 1 && Pz == 1
            train_function = @kbmf1k1k_semisupervised_classification_variational_train;
            test_function = @kbmf1k1k_semisupervised_classification_variational_test;
        elseif Px > 1 && Pz == 1
            train_function = @kbmf1mkl1k_semisupervised_classification_variational_train;
            test_function = @kbmf1mkl1k_semisupervised_classification_variational_test;
        elseif Px == 1 && Pz > 1
            train_function = @kbmf1k1mkl_semisupervised_classification_variational_train;
            test_function = @kbmf1k1mkl_semisupervised_classification_variational_test;
        elseif Px > 1 && Pz > 1
            train_function = @kbmf1mkl1mkl_semisupervised_classification_variational_train;
            test_function = @kbmf1mkl1mkl_semisupervised_classification_variational_test;
        end
    end

    for i = 1:2:nargin - 4
        parameters.(varargin{i}) = varargin{i + 1};
    end

    parameters.train_function = train_function;
    parameters.test_function = test_function;

    state = train_function(Kx, Kz, Y, parameters);
end