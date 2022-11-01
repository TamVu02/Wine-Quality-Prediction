%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

%Manipulate file
fid = fopen('train.csv','rt') ;
readFile = fread(fid) ;
fclose(fid) ;
readFile = char(readFile.') ;
fileOut =strrep(strrep(readFile, 'white', '1'), 'red', '0') ; % replace white with 1 and red with 0
fid2 = fopen('trainAlt.txt','wt') ;
fwrite(fid2,fileOut) ;
fclose (fid2) ;

%% Load Data
fprintf('Loading data ...\n');
data = dlmread('trainAlt.txt', ';', 1, 0); %except first line
quality=12;
X = data(1:end,[1:quality-1,quality+1:end]);
y = data(1:end,quality);
m = length(y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.003;
num_iters = 6000;

% Init Theta and Run Gradient Descent
theta = zeros(size(X,2), 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Compute test data ================
dlmwrite('result.csv', 'id, quality', 'delimiter', '');

%Manipulate file
fid = fopen('test.csv','rt') ;
readFile = fread(fid) ;
fclose(fid) ;
readFile = char(readFile.') ;
fileOut =strrep(strrep(readFile, 'white', '1'), 'red', '0') ; % replace white with 1 and red with 0
fid2 = fopen('test.csv','wt') ;
fwrite(fid2,fileOut) ;
fclose (fid2) ;

%compute quality
testData = dlmread('test.csv', ';', 1, 0);

normalizeInput=zeros(1, size(testData,2)-1);

%compute quality output
for i = 1 : size(testData,1)
  input=testData(i , 2:end);
  for j = 1 : size(normalizeInput,2)
    normalizeInput(1 , j) = (input(1,j) - mu(j)) / sigma(j);
  endfor
  quality=[1,normalizeInput]*theta;
  dlmwrite('result.csv', [testData(i , 1) , quality], ', ', '-append');
endfor





