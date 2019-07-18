function [eigenvalues, modes] = dmd_algo(X,Y,varargin)
% [eigenvalues, modes] = DMD(X,OPTIONS)
%
% OPTIONS:
% 'meth' = {1,2,3,4}
% 1: standard dmd algorithm
% 2: simple decomposition A=Y pinv(X)
% 3: algorithm 3 in Tu et al., 2013 (exact dmd)
% 4: algorithm 4 in Tu et al., 2013
% 'corr' = {0,1} : correction de-biasing
%
% default: 'meth'=1, 'corr'=0


% default values
method = 1;
correction = 0;

if ~isempty(varargin)
    
    for k = 1 : 2 : length(varargin)-1
        
        if strcmp(varargin{k},'meth')
            
            method = varargin{k+1};
            
        elseif strcmp(varargin{k},'corr')
            
            correction = varargin{k+1};
            
        end
        
    end
    
end

% correction de-biasing
if correction == 1
    [U_proj,S_proj,V_proj] = svd([X;Y]);
    n = size(X,1);
    X = X*V_proj(:,1:n)*V_proj(:,1:n)';
    Y = Y*V_proj(:,1:n)*V_proj(:,1:n)';
end


if method == 1
    % standard DMD algorithm
    [Uc,Sc,Vc] = svd(X,'econ');
%     r=12;
%     Uc=Uc(:,1:r);
%     Sc=Sc(1:r,1:r);
%     Vc=Vc(:,1:r);
    A_tilde = Uc'*Y*Vc/Sc;
    [w,eig_lambda] = eig(A_tilde);
    V = Uc*w;
    
elseif method == 2
    % simple decomposition A=Y pinv(X)
    A_ = Y/X;
    [V,eig_lambda] = eig(A_);

elseif method == 3
    % algorithm 3 in Tu et al., 2013 (exact DMD)
    [Uc,Sc,Vc] = svd(X,'econ');
    [Q,R] = qr([X Y]);
    A = Y*Vc*inv(Sc)*Uc';
    A_tildeQ = Q'*A*Q;
    [w,eig_lambda]=eig(A_tildeQ);
    V = Q*w;
    % % other way of computing the DMD/Koopman modes (algo 2 in Tu et al., 2013); corresponds to exact modes
    % V = Y*Vc*inv(Sc)*w*inv(eig_lambda);

elseif method == 4
    % algorithm 4 in Tu et al., 2013
    [Uc,Sc,Vc] = svd(X,'econ');
    p = X(:,end)-Uc*Uc'*X(:,end);
    q = p/norm(p);
    A_tilde = Uc'*Y*Vc/Sc;
    [w,eig_lambda] = eig(A_tilde);
    B = Y*Vc*inv(Sc);
    V = Uc*w+q*q'*B*w*eig_lambda;

end

% ordering
scaling = inv(eig_lambda)/V*X(:,2);
%scaling = pinv(V)*X(:,1);
[Y,I] = sort(abs(scaling),'descend');
V = V*diag(scaling);
modes = V(:,I);
eigenvalues = diag(eig_lambda(I,I));
