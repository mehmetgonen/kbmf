% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = kbmf_classification_test(Kx, Kz, state)
    directory = fileparts(mfilename('fullpath'));
    addpath([directory, '/kbmf1k1k']);
    addpath([directory, '/kbmf1k1mkl']);
    addpath([directory, '/kbmf1mkl1k']);
    addpath([directory, '/kbmf1mkl1mkl']);
    prediction = state.parameters.test_function(Kx, Kz, state);
end