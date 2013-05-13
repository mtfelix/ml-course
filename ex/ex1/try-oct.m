
%% Initialization
%clear ; close all; clc

fprintf('Plotting Data ...\n')
data = load('d.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples


X = [ones(m, 1), data(:,1)]; % Add a column of ones to x

theta = zeros(2, 1); % initialize fitting parameters


