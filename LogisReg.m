%% Initialization
clear ; close all; clc

%Manipulate file
fid = fopen('trainAlt.txt','rt') ;
readFile = fread(fid) ;
fclose(fid) ;
readFile = char(readFile.') ;
fileOut =strrep(strrep(readFile, ';6;', ';1;'), ';5;', ';0;') ; % replace quality 6 to 1 and 5 to 0
fid2 = fopen('trainAlt2.txt','wt') ;
fwrite(fid2,fileOut) ;
fclose (fid2) ;

%% Load Data
fprintf('Loading data ...\n');
data = dlmread('trainAlt2.txt', ';', 1, 0); %except first line
quality=12;
X = data(1:end,[1:quality-1,quality+1:end]);
y = data(1:end,quality);
m = length(y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 1: Regularized Logistic Regression ============

% Add Polynomial Features

%X = mapFeature(X);

%normalize Feature
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros)\n');
disp(grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = costFunctionReg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 1): %f\n', cost);
fprintf('Gradient at test theta :\n');
disp(grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

 disp(theta);

% Compute test dat
dlmwrite('result3.csv', 'id, quality', 'delimiter', '');

testData = dlmread('test.csv', ';', 1, 0);

normalizeInput=zeros(1, size(testData,2)-1);

for i = 1 : size(testData,1)
  input=testData(i , 2:end);
  for j = 1 : size(normalizeInput,2)
    normalizeInput(1 , j) = (input(1,j) - mu(j)) / sigma(j);
  endfor
  quality=5+predict(theta,normalizeInput);
  dlmwrite('result3.csv', [testData(i , 1) , quality], ', ', '-append');
endfor


