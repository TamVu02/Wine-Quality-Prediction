function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
updateTheta= zeros(size(X,2),1);

for iter = 1:num_iters
  h=0;
  %calculate h
  for i=1:size(X,2)
    h=ifelse (i==1,  (h+theta(i)) , (h+theta(i).*X(:,i)));
  endfor
  %calculate theta
  for i = 1:size(X,2)
    updateTheta(i,1)= ifelse(i==1 , (theta(i) - alpha * (1/m) * sum(h - y)) , (theta(i) - alpha * (1/m) * sum((h - y) .* X(:,i))));
  endfor
    %update theta
    theta = updateTheta;
    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);
end

end

