function R = get_vector(Cov)
% Feature transformation and vectorization of covariance matrix 
% Function: whitening, logarithmic transformation and vectorization of 
% covariance matrix

% Input: 
% Cov - covariance matrix (channels × channels × trials)

% Output: 
% R - Eigenvector matrix (trials × eigendimension) 
% Eigendimension = number of channels^2 (omnidirectional quantization)

[channels, ~, num_samples] = size(Cov);
feature_dim = channels * channels;
R = zeros(num_samples, feature_dim);

for s = 1:num_samples
    cov_mat = Cov(:, :, s);
    log_cov = logm(cov_mat);
    R(s, :) = log_cov(:)';
end
end