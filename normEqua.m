%% ================ Normal Equation ================
%% Load Data
fprintf('Loading data ...\n');
data = dlmread('trainAlt.txt', ';', 1, 0); %except first line
quality=12;
X = data(1:end,[1:quality-1,quality+1:end]);
y = data(1:end,quality);
m = length(y);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%compute quality
dlmwrite('result2.csv', 'id, quality', 'delimiter', '');

testData = dlmread('test.csv', ';', 1, 0);

for i = 1 : size(testData,1)
  input=testData(i , 2:end);
  quality=[1 , input]*theta;
  dlmwrite('result2.csv', [testData(i , 1) , quality], ', ', '-append');
endfor
