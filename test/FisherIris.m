clear
close all
% rng(0)

%% Init Test
tol = 0.1;
AssertClose = @(A, B, msg) assert(max(abs(A(:) - B(:))) < tol, msg);

%% Data Load and PreProcessing
load fisheriris % meas (150x4), species (150x1 cell)

Idx = 51:150;                       % versicolor vs virginica
X = meas(Idx, :);
Y = species(Idx);

YIdx = zeros(numel(Idx), 1);
YIdx(strcmp(Y, 'versicolor')) = 1;  % class 1
YIdx(strcmp(Y, 'virginica'))  = 2;  % class 2

N = numel(YIdx);
perm = randperm(N);
TrainRatio = 0.8;
nTrain = round(TrainRatio * N);

Train = perm(1:nTrain);
Test = perm(nTrain+1:end);

XTrain = X(Train, :);
XTest  = X(Test, :);
YTrain = YIdx(Train);
YTest  = YIdx(Test);

% Normalization
mu = mean(XTrain, 1);
sg = std(XTrain, 0, 1); sg(sg==0) = 1;
XTrain = (XTrain - mu) ./ sg;
XTest = (XTest - mu) ./ sg;

Ytr_oh = zeros(numel(YTrain), 2);
Ytr_oh(sub2ind(size(Ytr_oh), (1:numel(YTrain))', YTrain)) = 1;

XTrain = Tensor(XTrain);
YTrain = Tensor(Ytr_oh);
XTest = Tensor(XTest);

%% Model Creation
Mdl = Model();
Mdl = Mdl.Add(Dense(4, 2, true, 'xavier'));

opts = modelset('LearningRate', 0.1);
opt  = opts.OptimizerFcn(Mdl.Param(), opts);

%% Model Training
epochs = 1000;
LossHistory = zeros(epochs, 1);
for e = 1:epochs
    logits = Mdl.Forward(XTrain);
    Loss = CategoricalCrossEntropy(logits, YTrain);
    LossHistory(e) = Loss.Data;

    Mdl.ZeroGrad();
    Loss.Backward();
    opt.Step();
end

logits_te = Mdl.Forward(XTest);     % Computes the Test Set Logits
Pte = Softmax(logits_te).Data;      % Class Probabilities
[~, Pred] = max(Pte, [], 2);        % Argmax to get the Predicted Class

Accuracy = mean(Pred == YTest);
AssertClose(Accuracy, 1.0, 'Binary Iris accuracy too low');

fprintf('Fisher Iris Binary Classification Passed with Accuracy = %2.2d !\n', Accuracy);

%% Plot Learning
figure;
plot(1:epochs, LossHistory, 'LineWidth', 1.5);
grid on
xlabel('Epoch');
ylabel('Loss');
title('Training Loss')