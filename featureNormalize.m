function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

%calculate mean and sigma of each feature
for i = 1:size(X,2)
  mu(i)=mean(X(:,i));
  sigma(i)=std(X(:,i));
  X_norm(:,i)=(X(:,i)-mu(i))./sigma(i);
endfor
dlmwrite('normalizedData.txt',X_norm);


end
