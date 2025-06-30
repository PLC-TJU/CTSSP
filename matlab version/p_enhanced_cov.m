function [final_covariances, whiten_filter] = p_enhanced_cov(X, t_win, tau, whiten_filter)
% Calculate the enhanced covariance matrix (with whitening) 

% Input: 
% X: EEG data (n_channels, n_timepoints, n_trials) 
% t_win: list of time windows, where each element is [start, end] and the 
% units are points instead of time 
% tau: delay values (integer or integer array) 
% whiten_filter: whitening filter (training mode is empty) 
% estimator: covariance estimator (default 'cov') 
% metric: distance metric ('euclid' or 'riemann', default 'euclid')

% Output: 
% final_covariances: enhanced covariance matrix (channels × K × N,
% channels*K*N, trials). where K and N are the number of tau and t_win,
% respectively.
% whiten_filter: whitening filter (array of tuples, each element corresponds to a subwindow)

if nargin < 4, whiten_filter = []; end

[num_channels, num_time_points, num_samples] = size(X);
is_train = isempty(whiten_filter);
if is_train
    whiten_filter = {};
end

if isempty(tau)
    tau = 0;
end
K = numel(tau);

if isempty(t_win)
    t_win = {[1, num_time_points]};
elseif isnumeric(t_win) && numel(t_win) == 2
    t_win = {t_win(:)'};
end
N = numel(t_win);

enhanced_covs = cell(1, N);
for win_idx = 1:N
    win = t_win{win_idx};
    start_t = win(1);
    end_t = win(2);
    win_len = end_t - start_t + 1;
    
    if end_t > num_time_points
        error('Time window out of data range');
    end
    
    sub_signal = X(:, start_t:end_t, :);
    delayed_signals = cell(1, K);
    for k = 1:K
        d = tau(k);
        if d == 0
            delayed_signals{k} = sub_signal;
        else
            delayed = zeros(num_channels, win_len, num_samples);
            delayed(:, (d+1):end, :) = sub_signal(:, 1:(end-d), :);
            delayed_signals{k} = delayed;
        end
    end
    
    concat_signal = cat(1, delayed_signals{:}); % (num_channels*K, win_len, num_samples)
  
    Ne = size(concat_signal,1);
    Cov = zeros(Ne,Ne,num_samples);% (num_channels*K, num_channels*K, num_samples)
    for s = 1:num_samples
        Cov(:,:,s) = cov(concat_signal(:,:,s)');
    end
    
    for s = 1:num_samples
        tr_val = trace(Cov(:, :, s));
        if tr_val < 1e-10
            tr_val = 1e-10;
        end
        Cov(:, :, s) = Cov(:, :, s) / tr_val;
    end
    
    if is_train
        meanCov = mean(Cov, 3);
        [U, S] = eig(meanCov);
        W = U * diag(1./sqrt(diag(S))) * U';
        whiten_filter{win_idx} = W;
    else
        W = whiten_filter{win_idx};
    end
    
    whitened_cov = zeros(size(Cov));
    for s = 1:num_samples
        whitened_cov(:, :, s) = W' * Cov(:, :, s) * W;
    end
    enhanced_covs{win_idx} = whitened_cov;
end

total_dim = num_channels * K * N;
final_covariances = zeros(total_dim, total_dim, num_samples);

for s = 1:num_samples
    blocks = cell(1, N);
    for win_idx = 1:N
        blocks{win_idx} = enhanced_covs{win_idx}(:, :, s);
    end
    final_covariances(:, :, s) = blkdiag(blocks{:});
end
end