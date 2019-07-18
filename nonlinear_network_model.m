% dynamics used in example 2 in the submitted paper

function dX = nonlinear_network_model(t,X,param)

L = param.L;
deltaA = param.deltaA;

if size( X, 2) > 1
    X = X(:);
end

nPoints = length(X) / 2;
ix = (0 * nPoints + 1) : (1 * nPoints);
iy = (1 * nPoints + 1) : (2 * nPoints);


dX = [-1*X(ix)+deltaA(ix).*X(ix)-X(ix).^3-2*X(iy)+cos(X(iy)).*(L*(X(ix)+X(iy)+X(ix).^2));
        -1*X(iy)+deltaA(iy).*X(iy)+1*X(ix)-X(iy).^3+1.5*cos(X(iy)).*(L*(X(ix)+X(iy)+X(ix).^2))];