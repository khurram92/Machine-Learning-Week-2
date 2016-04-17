function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %disp(iter);
    theta = theta - (alpha/m) * X' * ((X * theta) - y);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %disp(J_history(iter));
    %break;
    if iter > 1,
        if J_history(iter) > J_history(iter-1),
            X = sprintf('Cost function is increasing, exit');
            disp(X)
            break
        elseif 100 * J_history(iter)/J_history(iter-1) > 99.999
            break
        end
    end
        %X = sprintf('%s is the cost after %d iterations', J_history(iter), iter);
    %disp(X)
end

%hold off;
i = 1:1:iter
plot(i, J_history(i));

pause;
return

end
