clear
close all
clc

rng(0)

%% Init Test
tol = 0.1;
AssertClose = @(A, B, msg) assert(max(abs(A(:) - B(:))) < tol, msg);

%% Data Load and PreProcessing
[XdTr, YdTr] = digitTrain4DArrayData;
[XdTe, YdTe] = digitTest4DArrayData;

X = permute(double(XdTr), [4 1 2 3]);
Y = cellstr(YdTr); Y = str2double(Y) + 1;

XTest = permute(double(XdTe), [4 1 2 3]);
YTest = cellstr(YdTe); YTest = str2double(YTest) + 1;

N = numel(Y);
perm = randperm(N);
TrainRatio = 1.0;
nTrain = round(TrainRatio * N);

Train = perm(1:nTrain);
Test  = 1:numel(YTest);

XTrain = X(Train, :, :, :);
YTrain = Y(Train);

XTrain = reshape(XTrain, [size(XTrain,1), 28*28]);
XTest  = reshape(XTest,  [size(XTest,1),  28*28]);

% Normalization
Mu = mean(XTrain, 1);
Std = std(XTrain, 0, 1); 
Std(Std == 0) = 1;

XTrain = (XTrain - Mu) ./ Std;
XTest  = (XTest  - Mu) ./ Std;

% One-Hots Labels
YTrainOH = zeros(numel(YTrain), 10);
YTrainOH(sub2ind(size(YTrainOH), (1:numel(YTrain))', YTrain)) = 1;

XTrain = Tensor(XTrain);
YTrain = Tensor(YTrainOH);
XTest  = Tensor(XTest);

%% Model Creation
Mdl = Model();
Mdl = Mdl.Add(Dense(28*28, 128, true,     'he',  'relu'));
Mdl = Mdl.Add(Dense(  128,  64, true,     'he',  'relu'));
Mdl = Mdl.Add(Dense(   64,  10, true, 'xavier'));   % logits (no activation)

opts = modelset('LearningRate', 0.01);
opt  = opts.OptimizerFcn(Mdl.Param(), opts);

%% Model Training
epochs = 200;
batch = 256;
LossHistory = zeros(epochs, 1);

Ntr = size(XTrain.Data, 1);
for e = 1:epochs
    Idx = randperm(Ntr);
    EpochLoss = 0.0;
    
    nb = 0;
    for s = 1:batch:Ntr
        t = min(s + batch - 1, Ntr);
        xb = XTrain(Idx(s:t), :);
        yb = YTrain(Idx(s:t), :);

        logits = Mdl.Forward(xb);
        Loss = CategoricalCrossEntropy(logits, yb);
        EpochLoss = EpochLoss + Loss.Data;
        nb = nb + 1;

        Mdl.ZeroGrad();
        Loss.Backward();
        opt.Step();
    end
    LossHistory(e) = EpochLoss / nb;
end

LogitsTest = Mdl.Forward(XTest);        % Model Logits
PTest = Softmax(LogitsTest).Data;       % Class Probability
[~, Pred] = max(PTest, [], 2);          % Argmax for Prediction Label

Accuracy = mean(Pred == YTest);
AssertClose(Accuracy, 1.0, 'MNIST accuracy too low');

fprintf('MNIST Dense(784x128x64x10 + ReLU) passed with Accuracy = %2.2f !\n', Accuracy);

%% Plot Learning
figure;
plot(1:epochs, LossHistory, 'LineWidth', 1.5);
grid on
xlabel('Epoch');
ylabel('Loss');
title('Training Loss')
