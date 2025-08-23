clear
close all
clc

rng(0)

%% Init Test
tol = 1e-3;
AssertClose = @(A, B, msg) assert(max(abs(A(:) - B(:))) < tol, msg);

%% Data Generation
N = 256;
D = 3;

coefs = Tensor(randi([-5, 5], [D, 1]));
bias  = Tensor(randi([-5, 5], [1, 1]));
X = Tensor(randn(N, D));
Y = X * coefs + bias;

%% Model Creation
Mdl = Model();
Mdl = Mdl.Add(Dense(D, 1, true, 'xavier'));

% Optimizer (SGD)
opts = modelset('LearningRate', 0.05);
opt  = opts.OptimizerFcn(Mdl.Param(), opts);

%% Train (full-batch)
epochs = 2000;
for epoch = 1:epochs
    Pred = Mdl.Forward(X);
    Loss = MeanSquaredError(Pred, Y);
    Mdl.ZeroGrad();
    Loss.Backward();
    opt.Step();
end

% Validate learned parameters
P = Mdl.Param();
W = P{1}.Data;   % Weight (D x 1)
B = P{2}.Data;   % Bias   (1 x 1)

AssertClose(W, coefs.Data, 'Linear regression: learned W is off');
AssertClose(B,  bias.Data, 'Linear regression: learned B is off');

disp('Linear Regression(Model Class) Passed All Tests!')
