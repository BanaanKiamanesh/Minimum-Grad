clear
close all
clc
rng(0)

addpath("../src/Actiavation/")

%% Init Test
tol = 1e-10;
AssertClose = @(A, B, msg) assert(max(abs(A(:) - B(:))) < tol, msg);

%% Sigmoid
x = Tensor(rand(3, 4) - 0.5);
y = Sigmoid(x);

AssertClose(y.Data, 1 ./ (1 + exp(-(x.Data))), 'Sigmoid forward wrong');

x = Tensor(rand(3, 4) - 0.5, true);
y = Sigmoid(x);
s = y.sum();
s.Backward();

Sig = 1 ./ (1 + exp(-(x.Data)));

AssertClose(x.Grad.Data, Sig .* (1 - Sig), 'Sigmoid grad wrong');

%% Tanh
x = Tensor(rand(2, 5) - 0.5);
y = Tanh(x);
AssertClose(y.Data, tanh(x.Data), 'Tanh forward wrong');

x = Tensor(rand(2, 5) - 0.5, true);
y = Tanh(x);
s = y.sum();
s.Backward();
tval = tanh(x.Data);

AssertClose(x.Grad.Data, 1 - tval .^ 2, 'Tanh grad wrong');

%% ReLU
xRaw = rand(4, 4) - 0.3;   % Shift to Get Both Signs
x = Tensor(xRaw, true);
y = Relu(x);
s = y.sum();
s.Backward();
Mask = xRaw > 0;

AssertClose(y.Data, max(xRaw, 0), 'ReLU forward wrong');
AssertClose(x.Grad.Data, double(Mask), 'ReLU grad wrong');

%% LeakyReLU
Leak = 0.07;

xRaw = rand(4, 3) - 0.3;
x = Tensor(xRaw, true);
y = LeakyRelu(x, Leak);
s = y.sum();
s.Backward();

yExp = max(xRaw, 0) + Leak*min(xRaw, 0);
gExp = double(xRaw > 0) + Leak*double(xRaw <= 0);

AssertClose(y.Data, yExp, 'LeakyReLU forward wrong');
AssertClose(x.Grad.Data, gExp, 'LeakyReLU grad wrong');

%% Softmax (2D, Along Last Dim)
X       = Tensor(rand(5, 6)-0.5);
Y       = Softmax(X);
RowSums = sum(Y.Data, 2);

AssertClose(RowSums, ones(5, 1), 'Softmax rows not summing to 1');
assert(all(Y.Data(:) >= 0 & Y.Data(:) <= 1), 'Softmax outputs not in [0, 1]');

% Softmax Gradient Test 1: Sum Over Rows -> Constant => Zero Grad
X = Tensor(rand(5, 6)-0.5, true);
Y = Softmax(X);
S = Y.sum();                 % Sums to Constant = Number of Rows
S.Backward();

AssertClose(X.Grad.Data, zeros(size(X.Data)), 'Softmax grad of constant sum should be zero');

% Softmax Gradient Test 2: General Upstream Gradient per Standard Jacobian
X = Tensor(rand(3, 4) - 0.5, true);
Y = Softmax(X);
G = Tensor(rand(3, 4));
Y.Backward(G);

Ynp = Y.Data;
Gnp = G.Data;

% Expected: for Each Row r, dL/dx = y .* (g - (g·y) )
ExpGrad = zeros(size(X.Data));
for r = 1:size(X.Data, 1)
    yr = Ynp(r, :)';
    gr = Gnp(r, :)';
    ExpGrad(r, :) = (yr .* (gr - (gr.' * yr)))';
end

AssertClose(X.Grad.Data, ExpGrad, 'Softmax explicit grad wrong');

disp('All tests passed!');
