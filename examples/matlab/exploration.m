% Matlab script to explore the data of our experiments. Particularly useful for
% debugging matrices.

clc; clear all; close all;
dirpath = '/home/alan/variable-projection/examples/matlab';
addpath(genpath(dirpath));

% read the Q matrix, which is at dirpath/Q.mtx
Qmatpath = fullfile(dirpath, 'bal-93/Q.mtx');
Q = mmread(Qmatpath);
Ltransmatpath = fullfile(dirpath, 'bal-93/Ltrans.mtx');
Ltrans = mmread(Ltransmatpath);


Ltransnorm = Ltrans + 1e-5 * speye(size(Ltrans));

figure(2);
N = null(full(Ltrans));
spy(N);

[labels, sizes, G] = components_from_laplacian(Ltrans);

%%
p = amd(Q);
L = chol(Q(p,p));

figure(1); clf;
subplot(2,2,1);
spy(Q);
n = size(Q, 1);
t = size(Ltrans,1);
% draw lines to show the blocks based on the size of Ltrans
vline = [n-t, n-t];
hline = [n-t, n-t];
hold on;
for i = 1:length(vline)
    x = vline(i);
    line([x x], ylim, 'Color', 'r', 'LineStyle', '--');
end
for i = 1:length(hline)
    y = hline(i);
    line(xlim, [y y], 'Color', 'r', 'LineStyle', '--');
end
hold off;
title('Sparsity pattern of Q');
subplot(2,2,2);
spy(Q(p,p));
title('Reordered sparsity pattern of Q');
subplot(2,2,3);
spy(L);
title('Sparsity pattern of Cholesky factor of Q(p,p)');
subplot(2,2,4);
% diff = L'*L - Q(p,p);
% cut anything very small
% diff(abs(diff) < 1e-8) = 0;
spy(Ltrans);
title('Sparsity pattern of Ltrans')


function [labels, sizes, G] = components_from_laplacian(L, tol)
%COMPONENTS_FROM_LAPLACIAN  Connected components from a Laplacian.
%   [labels, sizes, G] = components_from_laplacian(L, tol)
%   - L   : n-by-n Laplacian (combinatorial or normalized). Symmetric.
%           Off-diagonals should be <= 0 where edges exist.
%   - tol : (optional) threshold for treating tiny values as zero.
%           default = 1e-12 * max(1, norm(L,1))
%   Returns:
%     labels(i)  : component id (1..k) for node i
%     sizes(c)   : size of component c
%     G          : MATLAB graph object built from inferred adjacency
%
%   Notes:
%   * Works for weighted graphs; weights are taken as -L_ij (or scaled if normalized).
%   * Only the sign pattern matters for connectivity, so any positive scaling is fine.

    if nargin < 2 || isempty(tol)
        % scale tol to matrix magnitude; tweak if your L has large condition numbers
        tol = 1e-12 * max(1, norm(L, 1));
    end

    % Ensure sparse for efficiency
    if ~issparse(L), L = sparse(L); end

    % Symmetrize gently to remove tiny asymmetries
    L = 0.5 * (L + L.');

    % Build weight matrix from negative off-diagonals: W_ij = max(0, -L_ij), i!=j
    W = spfun(@(x) max(0, -x), L);           % turn negatives into positives, clamp others to 0
    W = W - spdiags(diag(W), 0, size(W,1), size(W,2));  % zero out diagonal
    % Drop tiny numerical crumbs
    W = spfun(@(x) (abs(x) > tol).*x, W);

    % Construct an undirected graph (matrix is symmetric already)
    G = graph(W, 'upper');  % avoid double-counting edges

    % Connected components
    [labels, sizes] = conncomp(G);  % labels in 1..k, sizes: 1-by-k
end
