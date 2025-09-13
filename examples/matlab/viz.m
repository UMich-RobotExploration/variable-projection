%% script to visualize the sparsity patterns of different matrices and save to file

clear; clc;
curdir = "/home/alan/variable-projection/examples/matlab/data";
% bal-392             IMC-gate        MipNerf-kitchen  mrclam6         Replica-REPoffice0_100  smallGrid3D                      sparsity_patterns_MIT.png      sparsity_patterns_parking-garage.png  sparsity_patterns_sphere2500.png  tinyGrid3D      TUM-room
% bal-93              IMC-temple      MipNerf-room     mrclam7         Replica-REPoffice1_100  sparsity_patterns_city10000.png  sparsity_patterns_mrclam2.png  sparsity_patterns_plaza1.png          sparsity_patterns_tiers.png       torus3D
% city10000           intel           MIT              parking-garage  Replica-REProom0_100    sparsity_patterns_grid3D.png     sparsity_patterns_mrclam4.png  sparsity_patterns_plaza2.png          sparsity_patterns_torus3D.png     TUM-computer-R
% factor_graph_small  M3500           mrclam2          plaza1          Replica-REProom1_100    sparsity_patterns_intel.png      sparsity_patterns_mrclam6.png  sparsity_patterns_single_drone.png    sphere2500                        TUM-computer-T
% grid3D              MipNerf-garden  mrclam4          plaza2          single_drone            sparsity_patterns_M3500.png      sparsity_patterns_mrclam7.png  sparsity_patterns_smallGrid3D.png     tiers                             TUM-desk
data_subdirs = {'bal-392', 'bal-93', 'city10000', 'factor_graph_small', ...
    'grid3D', 'intel', 'M3500', 'MipNerf-garden', 'MipNerf-kitchen', ...
    'MipNerf-room', 'mrclam2', 'mrclam4', 'mrclam6', 'mrclam7', ...
    'parking-garage', 'plaza1', 'plaza2', 'Replica-REPoffice0_100', ...
    'Replica-REPoffice1_100', 'Replica-REProom0_100', ...
    'Replica-REProom1_100', 'single_drone', 'sphere2500', ...
    'smallGrid3D', 'tiers', 'torus3D', 'TUM-computer-R', ...
    'TUM-computer-T', 'TUM-desk'};
% DataMatrix.mtx  OffDiag.mtx     Qmain.mtx
fnames = {'DataMatrix.mtx', 'OffDiag.mtx', 'Qmain.mtx'};

start = 1;
for i = 1:length(data_subdirs)
    if ~contains(data_subdirs{i}, 'garage')
        fprintf('Skipping %s ...\n', data_subdirs{i});
        continue;
    end
    filename = sprintf('sparsity_patterns_%s.png', data_subdirs{i});
    save_fpath = fullfile(curdir, filename);
    if isfile(save_fpath)
        fprintf('File %s already exists, skipping...\n', filename);
        % continue;
    end


    Qcc = mmread(fullfile(curdir, data_subdirs{i}, 'Qmain.mtx'));
    Q = mmread(fullfile(curdir, data_subdirs{i}, 'DataMatrix.mtx'));
    Qcf = mmread(fullfile(curdir, data_subdirs{i}, 'OffDiag.mtx'));
    Qcc_dims = size(Qcc, 1);
    Qff_dims = size(Q, 1) - size(Qcc, 1);
    Qff = Q(Qcc_dims+1:end, Qcc_dims+1:end);

    QffRed = Qff(1:end-1, 1:end-1); % remove last row and column to fix dimension
    p = amd(QffRed);
    QffRedPerm = QffRed(p, p);
    cholQff = chol(QffRedPerm, 'lower');

    % denseRes = QffRed * inv(QffRedPerm) * QffRed';
    % denseBlock = ones(2,2);

    % % want to visualize the sparsity patterns of Qcc, Q, Qcf, chol(Qff), and denseRes
    % figure(i - start + 1);

    % % Qcc
    % subplot(2,3,1);
    % spy(Qcc, 20);
    % sparsity_Qcc = nnz(Qcc) / numel(Qcc);
    % title(sprintf('Q_{cc}: %.2f%%', sparsity_Qcc*100))

    % % Q
    % subplot(2,3,2);
    % spy(Q, 20);
    % sparsity_Q = nnz(Q) / numel(Q);
    % title(sprintf('Q: %.2f%%', sparsity_Q*100))
    % subplot(2,3,3);

    % % Qcf
    % spy(Qcf, 20);
    % sparsity_Qcf = nnz(Qcf) / numel(Qcf);
    % title(sprintf('Q_{cf}: %.2f%%', sparsity_Qcf*100))

    % % cholQff
    % subplot(2,3,4);
    % spy(cholQff, 20);
    % sparsity_cholQff = nnz(cholQff) / numel(cholQff);
    % title(sprintf('chol(Q_{ff}): %.2f%%', sparsity_cholQff*100))

    % % denseRes
    % subplot(2,3,5);
    % spy(denseRes, 20);
    % sparsity_denseRes = nnz(denseRes) / numel(denseRes);
    % title(sprintf('Dense Product: %.2f%%', sparsity_denseRes*100))

    % % denseBlock
    % subplot(2,3,6);
    % spy(denseBlock, 20);
    % sparsity_denseBlock = nnz(denseBlock) / numel(denseBlock);
    % title(sprintf('Dense Block: %.2f%%', sparsity_denseBlock*100))

    % % Save
    % sgtitle(data_subdirs{i}, 'Interpreter', 'none');
    % saveas(gcf, fullfile(curdir, filename));
end

%%

figure(1);
msize=10;

% Qcc (1)
ax1 = subplot(2,2,1);
spy(Qcc, msize);
sparsity_Qcc = nnz(Qcc) / numel(Qcc);
title(sprintf('Q_{cc}: %.2f%%', sparsity_Qcc*100))

% Q (2)
ax2 = subplot(2,2,2);
spy(Q, msize);
sparsity_Q = nnz(Q) / numel(Q);
title(sprintf('Q: %.2f%%', sparsity_Q*100))

% Qcf (3)
ax3 = subplot(2,2,3);
spy(Qcf, msize);
sparsity_Qcf = nnz(Qcf) / numel(Qcf);
title(sprintf('Q_{cf}: %.2f%%', sparsity_Qcf*100))

% cholQff (4)
ax4 = subplot(2,2,4);
spy(cholQff, msize);
sparsity_cholQff = nnz(cholQff) / numel(cholQff);
title(sprintf('chol(Q_{ff}): %.2f%%', sparsity_cholQff*100))

% Save
sgtitle(data_subdirs{i}, 'Interpreter', 'none');
saveas(gcf, fullfile(curdir, filename));

%% save the subplots to different files

savedir = "/home/alan/ICRA2025-separable-structure/fig/matrices";

fig1 = figure('Visible', 'off');
copyobj(ax1, fig1);
title('');
axis off;
ax = gca; ax.Box = 'on';
exportgraphics(fig1, sprintf('%s/Qcc.png', savedir));

fig2 = figure('Visible', 'off');
copyobj(ax2, fig2);
title('');
axis off;
exportgraphics(fig2, sprintf('%s/Q.png', savedir));

fig3 = figure('Visible', 'off');
copyobj(ax3, fig3);
title('');
axis off;
exportgraphics(fig3, sprintf('%s/Qcf.png', savedir));

fig4 = figure('Visible', 'off');
copyobj(ax4, fig4);
title('');
axis off;
exportgraphics(fig4, sprintf('%s/cholQff.png', savedir));
