clear
close all
clc

rng(0)

%% Init Test
tol = 1e-3;
AssertClose = @(A, B, msg) assert(max(abs(A(:) - B(:))) < tol, msg);

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

%% Training Loop

for epoch = 1:150
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

% Validate Learned Parameters
AssertClose(w.Data, coef.Data, 'Linear regression: learned W is off');
AssertClose(b.Data, bias.Data, 'Linear regression: learned B is off');

disp('Linear Regression(Simple Implementation) Passed All Tests!')
