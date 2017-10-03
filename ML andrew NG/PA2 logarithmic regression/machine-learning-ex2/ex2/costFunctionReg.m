function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z=X*theta;
h_theta=(sigmoid(z)); % m x 1 is the m output predicted

%to unreqularize the first theta, convert it to zeros
theta2=theta;
theta2(1)=0;

J_unreg= (1/m).* sum( (-y.*log(h_theta)) - ((1-y).*log(1-h_theta)) );
J=J_unreg+ ( (lambda/(2*m))*sum(theta2.^2)); %all except first theta
%grad should be that of theta ie m x 1

errorval=h_theta-y;
grad_unreg=(1/m).*sum(errorval.*X);

grad=grad_unreg+((lambda/m).*theta2');
%check grad

% =============================================================

end
