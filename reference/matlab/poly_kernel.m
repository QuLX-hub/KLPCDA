%%Note: The original loop implementation of poly_kernel is not equivalent to (dot+1) ^ d.
%%The implementation has been changed to the standard and efficient (dot+1). ^ d. The original version can also be replaced,
%%but this is usually a more common definition.
%%Update date:2025/12/08
function K = poly_kernel(u,v,polyOrder,varargin)
% POLY_KERNEL  polynomial kernel (u*v' + 1)^d
% Usage: K = poly_kernel(X, Z, d)
if nargin < 3 || isempty(polyOrder)
    polyOrder = 3;
end
if ~isscalar(polyOrder) || ~isnumeric(polyOrder) || polyOrder~=floor(polyOrder) || polyOrder<1
    error('Bad polyOrder');
end

dotproduct = (u * v');       % m x n
K = (dotproduct + 1).^polyOrder;
end
