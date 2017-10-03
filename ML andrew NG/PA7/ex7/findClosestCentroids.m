function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

function l2=dist(a,b)
  l2=sum((a-b).^2) .^(0.5);
%    l2=norm(a-b);
end
for i=1:size(X,1)
  min=inf;
  for j=1:K
    if dist(X(i,:),centroids(j,:)) < min;
      min=dist(X(i,:),centroids(j,:));
      idx(i)=j;
    endif
  endfor
endfor
  
%other approach
%for i=1:length(X)
%
%    distance = inf;
%
%    for j=1:K
%
%        kDist = norm(X(i, :) - centroids(j, :));
%
%        if (kDist < distance)
%
%            distance = kDist;
%
%            idx(i) = j;
%
%        end
%
%    end
%
%end





% =============================================================

end

