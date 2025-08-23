clear
close all
clc
rng(0)

%% Init Test
tol = 1e-2;
AssertClose = @(A, B, msg) assert(max(abs(A(:) - B(:))) < tol, msg);

%% Data Generation
N = 256; D = 3;
coef = Tensor(randi([-5, 5], [D, 1]));
bias = Tensor(randi([-5, 5], [1, 1]));

X = Tensor(randn(N, D));
Y = X * coef + bias;

%% === SGD ===
Mdl = Model();
Mdl.Add(Dense(D, 1));
opts = modelset('Optimizer','sgd', 'LearningRate',0.05);
opt  = opts.OptimizerFcn(Mdl.Param(), opts);

for epoch = 1:2000
    Pred = Mdl.Forward(X);
    Loss = MeanSquaredError(Pred, Y);
    Mdl.ZeroGrad();
    Loss.Backward();
    opt.Step();
end

P = Mdl.Param(); W = P{1}.Data; B = P{2}.Data;
AssertClose(W, coef.Data, 'SGD: learned W is off');
AssertClose(B, bias.Data, 'SGD: learned B is off');
disp('SGD optimizer test passed.');

%% === Momentum ===
Mdl = Model();
Mdl.Add(Dense(D, 1));
opts = modelset('Optimizer','momentum', 'LearningRate',0.05, 'Momentum',0.9);
opt  = opts.OptimizerFcn(Mdl.Param(), opts);

for epoch = 1:1500
    Pred = Mdl.Forward(X);
    Loss = MeanSquaredError(Pred, Y);
    Mdl.ZeroGrad();
    Loss.Backward();
    opt.Step();
end

P = Mdl.Param(); W = P{1}.Data; B = P{2}.Data;
AssertClose(W, coef.Data, 'Momentum: learned W is off');
AssertClose(B, bias.Data, 'Momentum: learned B is off');
disp('Momentum optimizer test passed.');

%% === RMSprop ===
Mdl = Model();
Mdl.Add(Dense(D, 1));
opts = modelset('Optimizer','rmsprop', ...
    'LearningRate', 0.01, ...
    'Alpha',        0.95, ...  
    'Epsilon',      1e-8);
opt  = opts.OptimizerFcn(Mdl.Param(), opts);

for epoch = 1:3000
    Pred = Mdl.Forward(X);
    Loss = MeanSquaredError(Pred, Y);
    Mdl.ZeroGrad();
    Loss.Backward();
    opt.Step();
end

P = Mdl.Param(); W = P{1}.Data; B = P{2}.Data;
AssertClose(W, coef.Data, 'RMSprop: learned W is off');
AssertClose(B, bias.Data, 'RMSprop: learned B is off');
disp('RMSprop optimizer test passed.');


%% === Adam ===
Mdl = Model();
Mdl.Add(Dense(D, 1));
opts = modelset('Optimizer','adam', ...
    'LearningRate', 0.01, ...  
    'Beta1',        0.9, ...
    'Beta2',        0.999, ...
    'Epsilon',      1e-8);
opt  = opts.OptimizerFcn(Mdl.Param(), opts);

for epoch = 1:3000       
    Pred = Mdl.Forward(X);
    Loss = MeanSquaredError(Pred, Y);
    Mdl.ZeroGrad();
    Loss.Backward();
    opt.Step();
end

P = Mdl.Param(); W = P{1}.Data; B = P{2}.Data;
AssertClose(W, coef.Data, 'Adam: learned W is off');
AssertClose(B, bias.Data, 'Adam: learned B is off');
disp('Adam optimizer test passed.');
