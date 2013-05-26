function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% the following implementation learned from prediction.m in ex4 
m = size(X, 1);
% activation in layer 2
a2 = sigmoid([ones(m, 1) X] * Theta1');
% hypo = a3  
h = sigmoid([ones(m, 1) a2] * Theta2');

%% extends y from (5000 x 1) to (5000 x 10)
% 1. simple way
 y3=(y==[1:num_labels]);
% 2. stepping way
%y1 = ones(5000,1)*(1:num_labels);
%y2 = y * ones(1,num_labels);
%y3 = (y1 == y2);

% the calculation of J is same to lrCostFunction in ex3 without regularization
%% Since the h and y are matrix not just vector, we need to sum two times, once for col once for row
%% 
J = (1/m) * sum( sum(((-1) .* y3) .* log(h) - ( 1 .- y3) .* log(1 .- h) ));

regularization  =  0;
Theta1_sq_sum = sum( (Theta1 .** 2)(:) );
Theta2_sq_sum = sum( (Theta2 .** 2)(:) );

% Note that you should not be regularizing the terms that correspond to the bias.
Theta1_bias_sq_sum  = sum(Theta1 .** 2)(:,1);
Theta2_bias_sq_sum  = sum(Theta2 .** 2)(:,1);

regularization = lambda/(2*m) * (Theta1_sq_sum + Theta2_sq_sum - Theta1_bias_sq_sum - Theta2_bias_sq_sum);

J = J + regularization;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
m = size(X, 1);
XX = [ones(m, 1) X];

for t =  1:m

    %%% step 1
	
    % a1 is 1x401
	a1 = XX(t,:);
	
	% Theta1 is 25 x 401
	% z2 is 1 x 25 
	z2 = a1 * Theta1' ;
	
	% a2 is 1 x 26 (add a0, the bias)
	a2 = sigmoid(z2);
	a2 = [ones(size(a2,1),1) a2];
	
	% Theta2 is 10 x 26
	% a3 is 1 x 10, a3 is the h
	z3 = a2 * Theta2';
	a3 = sigmoid(z3);
	
	%%% step 2
	
	% y3 is 5000 x 10, y3(t) stands for the y-vector of t-th sample. 
	delta3  = ( a3 .- y3(t) );
	
endfor






% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
