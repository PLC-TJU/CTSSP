% Modified from SBLEST
function [W, alpha, V, features] = sbl_kernel(R, Y)
% Sparse Bayesian learning
% Input
% R : Feature matrix (trials × feature dimension).
% Y : Label vector (trials × 1).

% Output 
% W : Low rank projection matrix. 
% alpha : Feature weight vector. 
% V : Feature vector matrix (each column is a time-frequency-space filter). 
% features : Optimized feature matrix (trials × feature dimension).

% Check properties of R
[M, DR] = size(R); 
Dim = round(sqrt(DR));
Loss_old = 1e12;
threshold = 0.05; 
maxiter = 5000;

% Check if R is symmetric
for c = 1:M
    row_cov = reshape(R(c,:), Dim, Dim);
    if ( norm(row_cov - row_cov','fro') > 1e-4 )
        disp('ERROR: Measurement row does not form symmetric matrix');
        return
    end
end

% Initializations
U = zeros(Dim, Dim);
Psi = eye(Dim); 
lambda = 1;

% Optimization loop
for i = 1:maxiter
    
    % Update U
    RPR = zeros(M, M);
    B = zeros(Dim^2, M);
    for c = 1:Dim
        start = (c-1)*Dim + 1; stop = start + Dim - 1;
        Temp = Psi*R(:, start:stop)';
        B(start:stop,:) = Temp;
        RPR =  RPR + R(:, start:stop)*Temp;
    end
    Sigma_y = RPR + lambda*eye(M);
    uc = B*(Sigma_y\Y );
    Uc = reshape(uc, Dim, Dim);
    U = (Uc + Uc')/2; 
    u = U(:);

    % Update Phi (dual variable of Psi)
    Phi = cell(1, Dim);
    SR = Sigma_y\R;
    for c = 1:Dim
        start = (c-1)*Dim + 1; stop = start + Dim - 1;
        Phi{1,c} = Psi - Psi * ( R(:,start:stop)' * SR(:,start:stop) ) * Psi;
    end
    
    % Update Psi
    PHI = 0;
    UU = 0;
    for c = 1:Dim
        PHI = PHI +  Phi{1, c};
        UU = UU + U(:,c) * U(:,c)';
    end
    Psi = ((UU + UU')/2 + (PHI + PHI')/2 )/Dim;
    
    % Update theta (dual variable of lambda) and lambda
    theta = 0;
    for c = 1:Dim
        start = (c-1)*Dim + 1; stop = start + Dim - 1;
        theta = theta +trace(Phi{1,c}* R(:,start:stop)'*R(:,start:stop)) ;
    end
    lambda = (sum((Y-R*u).^2) + theta)/M;
    
    % Convergence check
    logdet_Sigma_y =  calculate_log_det(Sigma_y);
    Loss = Y'*Sigma_y^(-1)*Y + logdet_Sigma_y;
    delta_loss = abs(Loss_old-Loss)/abs( Loss_old);
    if (delta_loss < 1e-4)
        break;
    end
    Loss_old = Loss;
end

% Eigendecomposition of W
W = U;
[~, D, V_all] = eig(W);
alpha_all = diag(D);

% Determine spatio-temporal filters V and classifier weights alpha
d = abs(diag(D)); 
d_max = max(d);
w_norm = d/d_max; 
index = find(w_norm > threshold);
V = V_all(:,index); 
alpha = alpha_all(index);

features = zeros(size(R,1), size(V,2));
for i = 1:size(R,1)
    sample_mat = reshape(R(i,:), sqrt(size(R,2)), []);
    features(i,:) = diag(V'*sample_mat*V);
end

end