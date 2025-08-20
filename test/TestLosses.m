clear
close all
clc
rng(0)

addpath("../src/")

%% Init Test

tol = 1e-10;
AssertClose = @(A, B, msg) assert(max(abs(A(:) - B(:))) < tol, msg);

%% Mean Squared Error — Forward (scalar loss)
x = Tensor(rand(3, 4));            % preds
y = Tensor(rand(3, 4));            % actual
L = MeanSquaredError(x, y);

Expect = mean((x.Data - y.Data).^2, 'all');   % numeric MSE
AssertClose(L.Data, Expect, 'MSE forward wrong');

%% Mean Squared Error — Backward Gradient wrt Preds
x = Tensor(rand(5, 6), true);       % preds (requires grad)
y = Tensor(rand(5, 6));             % actual
L = MeanSquaredError(x, y);
L.Backward();

N = numel(x.Data);
grad_expect = (2 / N) .* (x.Data - y.Data);

AssertClose(x.Grad.Data, grad_expect, 'MSE grad wrt preds wrong');

%% Categorical Cross-Entropy — Forward (Scalar Loss)
N = 4; C = 5;
logits = Tensor(rand(N, C) - 0.5);      % logits (no grad for forward check)
labels = zeros(N, C);

idx = randi(C, N, 1);
for n = 1:N
    labels(n, idx(n)) = 1;
end

y = Tensor(labels);

L = CategoricalCrossEntropy(logits, y);

% numeric softmax + CE
Z = logits.Data;
Z = Z - max(Z, [], 2);
expZ = exp(Z);
P = expZ ./ sum(expZ, 2);

Expect = -mean(sum(labels .* log(P), 2));

AssertClose(L.Data, Expect, 'CCE forward wrong');

%% Categorical Cross-Entropy — backward gradient wrt logits
N = 3; C = 4;
Z = Tensor(rand(N, C) - 0.2, true);      % Logits (Requires Grad)
labels = zeros(N, C);

idx = randi(C, N, 1);
for n = 1:N
    labels(n, idx(n)) = 1;
end

Y = Tensor(labels);

L = CategoricalCrossEntropy(Z, Y);
L.Backward();

% numeric grad: (softmax(Z) - Y) / N
Znum = Z.Data;
Znum = Znum - max(Znum, [], 2);
expZ = exp(Znum);
P = expZ ./ sum(expZ, 2);
grad_expect = (P - labels) ./ N;

AssertClose(Z.Grad.Data, grad_expect, 'CCE grad wrt logits wrong');

disp('All Tests Passed!');
