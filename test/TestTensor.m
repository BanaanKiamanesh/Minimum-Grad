clear
close all
% clc
rng(0)

addpath("../src/")
%% Init Test

tol = 1e-10;
AssertClose = @(A, B, msg) assert(max(abs(A(:) - B(:))) < tol, msg);

%% Initializers

sz = [3, 4];

T0 = Tensor(sz, [], [], 'Init', 'zero');
assert(all(T0.Data(:) == 0), 'zero init wrong');

T1 = Tensor(sz, [], [], 'Init', 'one');
assert(all(T1.Data(:) == 1), 'one init wrong');

sz = [1000, 50];
Th = Tensor(sz, [], [], 'Init', 'he');
Tx = Tensor(sz, [], [], 'Init', 'xavier');

FanIn = sz(1);
STDh = std(Th.Data(:));
STDx = std(Tx.Data(:));
Exph = sqrt(2 / FanIn);
Expx = sqrt(1 / FanIn);

assert(abs(STDh - Exph) < 0.1 * Exph, 'he init std wrong');
assert(abs(STDx - Expx) < 0.1 * Expx, 'xavier init std wrong');

T2 = Tensor([5, 6], [], [], 'Init', 'one');
assert(isequal(size(T2.Data), [5, 6]), 'init size wrong');

%% Basic Ops
% sum
x = Tensor(rand(2, 3), true);
sz = x.sum();
sz.Backward();

AssertClose(x.Grad.Data, ones(2, 3), 'sum/backprop failed');

% log
x = Tensor(rand(2, 3) + 1.0, true);
y = x.log();
y = y.sum();
y.Backward();

AssertClose(x.Grad.Data, 1 ./ x.Data, 'log grad wrong');

% exp
x = Tensor(rand(2, 3), true);
y = x.exp();
y = y.sum();
y.Backward();

AssertClose(x.Grad.Data, exp(x.Data), 'exp grad wrong');

% uminus
x = Tensor(rand(2, 2), true);
y = -x;
y = y.sum();
y.Backward();

AssertClose(x.Grad.Data, -ones(2, 2), 'uminus grad wrong');

%%  Addition & Subtraction with Broadcasting
a = Tensor(rand(2, 3), true);
b = Tensor(rand(1, 3), true);
z = a + b;
z = z.sum();
z.Backward();

AssertClose(a.Grad.Data,   ones(2, 3), '+ grad (a) wrong');
AssertClose(b.Grad.Data, 2*ones(1, 3), '+ grad (b) wrong');

a = Tensor(rand(2, 3), true);
b = Tensor(rand(1, 3), true);
z = a - b;
z = z.sum();
z.Backward();

AssertClose(a.Grad.Data,    ones(2, 3), '- grad (a) wrong');
AssertClose(b.Grad.Data, -2*ones(1, 3), '- grad (b) wrong');

%% Edge Case: Column Vector Broadcast
a = Tensor(rand(4, 5), true);
b = Tensor(rand(4, 1), true);
z = a + b;
z = z.sum();
z.Backward();

AssertClose(b.Grad.Data, sum(ones(4, 5), 2), 'broadcast col grad wrong');

%%  Multiplication (Elementwise)
a = Tensor(rand(2, 3), true);
b = Tensor(rand(1, 3), true);
z = a .* b;
z = z.sum();
z.Backward();

AssertClose(a.Grad.Data, repmat(b.Data, 2, 1), '.* grad (a) wrong');
AssertClose(b.Grad.Data,       sum(a.Data, 1), '.* grad (b) wrong');

%%  Matrix Multiply
A = Tensor(rand(2, 3), true);
B = Tensor(rand(3, 4), true);
S = A * B;
S = S.sum();
S.Backward();

expectA = ones(2, 4) * B.Data.';
expectB = A.Data.' * ones(2, 4);

AssertClose(A.Grad.Data, expectA, '* grad (A) wrong');
AssertClose(B.Grad.Data, expectB, '* grad (B) wrong');

%% Rectangular MatMul
A = Tensor(rand(5, 2), true);
B = Tensor(rand(2, 1), true);
S = A * B;
S = S.sum();
S.Backward();

AssertClose(A.Grad.Data, repmat(B.Data.', 5, 1), 'rect mmul grad A wrong');
AssertClose(B.Grad.Data,       sum(A.Data, 1).', 'rect mmul grad B wrong');

%%  Power
x = Tensor(rand(2, 3), true);
p = 3;
z = x.^p;
z = z.sum();
z.Backward();

AssertClose(x.Grad.Data, p * (x.Data .^ (p-1)), '.^ grad wrong');

%% Reciprocal
x = Tensor(rand(2, 3) + 1, true);
z = x.^-1;
z = z.sum();
z.Backward();

AssertClose(x.Grad.Data, -1 .* (x.Data .^ -2), 'reciprocal grad wrong');

%%  Slicing
x = Tensor(reshape(1:12, 3, 4), true);
y = x(2:3, 2:4);
y = y.sum();
y.Backward();

expect = zeros(3, 4);
expect(2:3, 2:4) = 1;

AssertClose(x.Grad.Data, expect, 'slice grad wrong');

%% Slice Single Element
x = Tensor(magic(3), true);
y = x(2, 2);
y.Backward();

expect = zeros(3, 3);
expect(2, 2) = 1;

AssertClose(x.Grad.Data, expect, 'single slice grad wrong');

%%  Explicit Grads
x = Tensor(rand(2, 2), true);
y = x + 1;
g = Tensor([1 2; 3 4]);
y.Backward(g);

AssertClose(x.Grad.Data, g.Data, 'explicit grad wrong');

%%  Numeric + Tensor Mixing
x = Tensor(rand(2, 3), true);
y = 3 + x;
y = y.sum();
y.Backward();

AssertClose(x.Grad.Data, ones(2, 3), 'numeric + Tensor grad wrong');

x = Tensor(rand(2, 3), true);
y = x + 3;
y = y.sum();
y.Backward();

AssertClose(x.Grad.Data, ones(2, 3), 'Tensor + numeric grad wrong');

%%  Chain Rule Sanity
x = Tensor(rand(2, 2), true);
y = x .* x;
y = y.sum();
y.Backward();

AssertClose(x.Grad.Data, 2 .* x.Data, 'chain rule 2x failed');

disp('Tensor Operations Passed All Tests!');
