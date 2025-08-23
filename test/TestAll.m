clear
close all
clc

addpath("../src/Actiavation/")
addpath("../src/Layers/")
addpath("../src/Loss/")
addpath("../src/Optimizers/")

%% Run All Test Files

tic;

TestTensor;
TestLosses;
TestActivation;
TestOptimizers;

fprintf('\n');
toc; 

%% Clear Mem and Close Plots
clear;
close all;
