clear
close all
rng(0)

%% Data Generation
% Actual Coefs
coef = Tensor(randi(10, [3, 1]));
bias = Tensor(randi(10));

% Data
X = Tensor(randn(100, 3));
Y = X * coef + bias;

% Initial Values for Bias and Weight
w = Tensor(randn(3, 1), true);
b = Tensor(randn(1, 1), true);

%% Learning Params
lr = 0.01;              % Learning Weight
bs = 20;                % Batch Size

for epoch = 1:100
    EpochLoss = 0;
    for Start = 1:bs:100
        End = Start + bs - 1;

        In     = X(Start:End, :);
        Pred   = In * w + b;
        Actual = Y(Start:End, :);

        Loss = MeanSquaredError(Pred, Actual);
        Loss.Backward();

        EpochLoss = EpochLoss + Loss.Data;

        % Gradient Descent
        w = w - (w.Grad .* Tensor(lr));
        b = b - (b.Grad .* Tensor(lr));
    end
end

fprintf("\n------------------------------------------------\n");
fprintf("Test on Linear Regression... \n");

fprintf('Actual w: %s \n', mat2str(coef.Data'));
fprintf('Estimated w: %s \n', mat2str(round(w.Data', 3)));

fprintf('Actual b: %d \n', bias.Data');
fprintf('Estimated b: %d \n', round(b.Data', 3));

fprintf("\n------------------------------------------------\n");