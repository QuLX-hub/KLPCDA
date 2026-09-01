function KLPCDA_main(datasetID, methodID)
% KLPCDA_main Minimal runnable implementation for KLPCDA Methods 1..7
% Usage: KLPCDA_main(datasetID, methodID)
% datasetID: 1=IndianPines, 2=CWRU, 3=GSE44076, 4=JAFFE_ID, 5=JAFFE_EXP
% methodID: 1..7
%
% Expect folder structure:
% data/<DatasetName>/traindata01.mat ... testlabels10.mat
% each .mat must contain variable named exactly traindata/trainlabels/...
%
% This file keeps everything in one script (no external helpers except poly_kernel.m).
%
% Author: (Qu Lingxiao)
% Date: 2025-12-08

if nargin<2
    error('Usage: KLPCDA_main(datasetID, methodID)');
end
if methodID <1 || methodID>7
    error('methodID must be 1..7');
end

% ---------------------------
% dataset mapping & paths
% ---------------------------
datasetNames = {'IndianPines','CWRU','GSE44076','JAFFE_ID','JAFFE_EXP'};
if datasetID <1 || datasetID > length(datasetNames)
    error('Unknown datasetID');
end
dataDir = fullfile('data', datasetNames{datasetID});
fprintf('Running KLPCDA: dataset=%s, method=%d\n', datasetNames{datasetID}, methodID);

% ---------------------------
% basic settings
% ---------------------------
numGroups = 10;  % expected 10 folds
dValues = 1:30;  % polynomial degrees to search
% p ranges per dataset (use the table from paper)
switch datasetID
    case 1; pValues = 1:105;    % Indian Pines
    case 2; pValues = 1:8;      % CWRU
    case 3; pValues = 1:15;     % GSE44076
    case {4,5}; pValues = 1:60; % JAFFE variants
    otherwise; pValues = 1:30;
end
p1_range = 0.1:0.1:100;  % fusion coefficient grid
p2_range = 0.1:0.1:100;  % used only for methods with 2 coeffs

% ---------------------------
% load folds safely
% ---------------------------
% each fold file: traindata%02d.mat (var traindata), trainlabels%02d.mat (var trainlabels), ...
folds = repmat(struct('traindata',[],'trainlabels',[],'valdata',[],'vallabels',[],'testdata',[],'testlabels',[]),1,numGroups);
expectedFields = {'traindata','trainlabels','valdata','vallabels','testdata','testlabels'};
for g = 1:numGroups
    for f = 1:numel(expectedFields)
        fname = expectedFields{f};
        fn = fullfile(dataDir, sprintf('%s%02d.mat', fname, g));
        if exist(fn,'file')
            S = load(fn);
            if isfield(S, fname)
                folds(g).(fname) = S.(fname);
            else
                warning('File %s exists but variable %s missing. Filling []', fn, fname);
                folds(g).(fname) = [];
            end
        else
            warning('Missing file: %s. Filling []', fn);
            folds(g).(fname) = [];
        end
    end
end

% quick sanity check: at least one fold must have non-empty data
if all(cellfun(@(x) isempty(x), arrayfun(@(s) s.traindata, folds, 'UniformOutput',false)))
    error('No training data loaded. Check dataDir and file names/variables.');
end

% ---------------------------
% Stage 1: grid search over (d,p) using validation sets
% We'll compute mean validation OA across folds for each (d,p)
% ---------------------------
fprintf('Stage 1: grid search (d,p) ...\n');
val_OA = zeros(length(dValues), length(pValues));

for di = 1:length(dValues)
    d = dValues(di);
    for pi = 1:length(pValues)
        p = pValues(pi);
        accs = zeros(numGroups,1);
        for g=1:numGroups
            T = folds(g);
            if isempty(T.traindata) || isempty(T.valdata); accs(g)=NaN; continue; end

            Xtr = double(T.traindata); ytr = double(T.trainlabels(:));
            Xv  = double(T.valdata);   yv  = double(T.vallabels(:));
            n = size(Xtr,1);
            if n==0; accs(g)=NaN; continue; end

            % build class matrices W and B
            classes = unique(ytr); L = numel(classes);
            n_L = arrayfun(@(c) sum(ytr==c), classes);
            % P diagonal blocks
            P_diag = [];
            ones_diag = [];
            for i=1:L
                P_diag = blkdiag(P_diag, (n_L(i)/n)*eye(n_L(i)));
                ones_diag = blkdiag(ones_diag, (1/n)*ones(n_L(i)));
            end
            W = P_diag - ones_diag;
            D = [];
            for i=1:L
                D = blkdiag(D, (1/n_L(i))*ones(n_L(i)));
            end
            J = ones(n);
            B = (1/n)*D - (1/n^2)*J;

            % compute kernels
            Ktr = center_kernel( poly_kernel(Xtr, Xtr, d) );
            Kval = center_kernel_cross( poly_kernel(Xv, Xtr, d), Ktr );

            % build E according to methodID, but during stage1 we use unit fusion weights:
            E = method_E_builder(methodID, 1, 1, W, B, Ktr, n);

            % eigen-decomp and projection
            [U, ~] = eig(E);
            [~,I] = sort(diag(real(U'* (Ktr) * U)),'descend'); % sort by projected energy proxy
            U = U(:,I);
            % take first p eigenvectors and normalize as in your code
            p_use = min(p, size(U,2));
            Uproj = zeros(n, p_use);
            for ii=1:p_use
                u = real(U(:,ii));
                denom = sqrt(max(u'*Ktr*u, eps));
                Uproj(:,ii) = u ./ denom;
            end
            Ytrain = Ktr * Uproj;
            Yval   = Kval * Uproj;

            mdl = fitcknn(Ytrain, ytr, 'NumNeighbors', 1);
            pred = predict(mdl, Yval);
            accs(g) = sum(pred==yv)/numel(yv);
        end
        val_OA(di,pi) = nanmean(accs);
    end
    fprintf(' d=%2d done\n', d);
end

% pick best d,p (argmax)
[best_val, idx] = max(val_OA(:));
[best_d_idx, best_p_idx] = ind2sub(size(val_OA), idx);
best_d = dValues(best_d_idx);
best_p = pValues(best_p_idx);
fprintf('Best (stage1) found: d=%d, p=%d, val_OA=%.4f\n', best_d, best_p, best_val);

% ---------------------------
% Stage 2: search fusion coefficient(s) using validation
% Method 1&2 have p1 & p2; Methods 3..7 have a single p1.
% ---------------------------
fprintf('Stage 2: fusion coefficient search ...\n');
best_p1 = 1; best_p2 = 1; best_val2 = -inf;
if methodID==1 || methodID==2
    for p1 = p1_range
        for p2 = p2_range
            accs = zeros(numGroups,1);
            for g=1:numGroups
                T = folds(g);
                if isempty(T.traindata) || isempty(T.valdata); accs(g)=NaN; continue; end
                Xtr = double(T.traindata); ytr = double(T.trainlabels(:));
                Xv  = double(T.valdata);   yv  = double(T.vallabels(:));
                n = size(Xtr,1);
                classes = unique(ytr); L = numel(classes);
                n_L = arrayfun(@(c) sum(ytr==c), classes);
                P_diag = []; ones_diag = [];
                for i=1:L
                    P_diag = blkdiag(P_diag, (n_L(i)/n)*eye(n_L(i)));
                    ones_diag = blkdiag(ones_diag, (1/n)*ones(n_L(i)));
                end
                W = P_diag - ones_diag;
                D = [];
                for i=1:L; D = blkdiag(D, (1/n_L(i))*ones(n_L(i))); end
                J = ones(n); B = (1/n)*D - (1/n^2)*J;

                Ktr = center_kernel( poly_kernel(Xtr, Xtr, best_d) );
                Kval = center_kernel_cross( poly_kernel(Xv, Xtr, best_d), Ktr );

                E = method_E_builder(methodID, p1, p2, W, B, Ktr, n);
                [U, ~] = eig(E);
                [~,I] = sort(diag(real(U'* (Ktr) * U)),'descend');
                U = U(:,I);
                p_use = min(best_p, size(U,2));
                Uproj = zeros(n, p_use);
                for ii=1:p_use
                    u = real(U(:,ii));
                    denom = sqrt(max(u'*Ktr*u, eps));
                    Uproj(:,ii) = u ./ denom;
                end
                Ytrain = Ktr * Uproj;
                Yval   = Kval * Uproj;
                mdl = fitcknn(Ytrain, ytr, 'NumNeighbors', 1);
                pred = predict(mdl, Yval);
                accs(g) = sum(pred==yv)/numel(yv);
            end
            meanacc = nanmean(accs);
            if meanacc > best_val2
                best_val2 = meanacc;
                best_p1 = p1; best_p2 = p2;
            end
        end
    end
else
    for p1 = p1_range
        accs = zeros(numGroups,1);
        for g=1:numGroups
            T = folds(g);
            if isempty(T.traindata) || isempty(T.valdata); accs(g)=NaN; continue; end
            Xtr = double(T.traindata); ytr = double(T.trainlabels(:));
            Xv  = double(T.valdata);   yv  = double(T.vallabels(:));
            n = size(Xtr,1);
            classes = unique(ytr); L = numel(classes);
            n_L = arrayfun(@(c) sum(ytr==c), classes);
            P_diag = []; ones_diag = [];
            for i=1:L
                P_diag = blkdiag(P_diag, (n_L(i)/n)*eye(n_L(i)));
                ones_diag = blkdiag(ones_diag, (1/n)*ones(n_L(i)));
            end
            W = P_diag - ones_diag;
            D = [];
            for i=1:L; D = blkdiag(D, (1/n_L(i))*ones(n_L(i))); end
            J = ones(n); B = (1/n)*D - (1/n^2)*J;

            Ktr = center_kernel( poly_kernel(Xtr, Xtr, best_d) );
            Kval = center_kernel_cross( poly_kernel(Xv, Xtr, best_d), Ktr );

            E = method_E_builder(methodID, p1, 1, W, B, Ktr, n);
            [U, ~] = eig(E);
            [~,I] = sort(diag(real(U'* (Ktr) * U)),'descend');
            U = U(:,I);
            p_use = min(best_p, size(U,2));
            Uproj = zeros(n, p_use);
            for ii=1:p_use
                u = real(U(:,ii));
                denom = sqrt(max(u'*Ktr*u, eps));
                Uproj(:,ii) = u ./ denom;
            end
            Ytrain = Ktr * Uproj;
            Yval   = Kval * Uproj;
            mdl = fitcknn(Ytrain, ytr, 'NumNeighbors', 1);
            pred = predict(mdl, Yval);
            accs(g) = sum(pred==yv)/numel(yv);
        end
        meanacc = nanmean(accs);
        if meanacc > best_val2
            best_val2 = meanacc;
            best_p1 = p1;
        end
    end
end
fprintf('Best fusion coeffs: p1=%.2f, p2=%.2f (val OA=%.4f)\n', best_p1, best_p2, best_val2);

% ---------------------------
% Final evaluation on test sets using chosen best_d, best_p, best_p1/p2
% ---------------------------
fprintf('Final evaluation on test folds ...\n');
all_preds = []; all_labels = [];
accs = zeros(numGroups,1);

for g=1:numGroups
    T = folds(g);
    if isempty(T.traindata) || isempty(T.testdata); accs(g)=NaN; continue; end
    Xtr = double(T.traindata); ytr = double(T.trainlabels(:));
    Xte = double(T.testdata);  yte = double(T.testlabels(:));
    n = size(Xtr,1);
    classes = unique(ytr); L = numel(classes);
    n_L = arrayfun(@(c) sum(ytr==c), classes);
    P_diag = []; ones_diag = [];
    for i=1:L
        P_diag = blkdiag(P_diag, (n_L(i)/n)*eye(n_L(i)));
        ones_diag = blkdiag(ones_diag, (1/n)*ones(n_L(i)));
    end
    W = P_diag - ones_diag;
    D = [];
    for i=1:L; D = blkdiag(D, (1/n_L(i))*ones(n_L(i))); end
    J = ones(n); B = (1/n)*D - (1/n^2)*J;

    Ktr = center_kernel( poly_kernel(Xtr, Xtr, best_d) );
    Kte = center_kernel_cross( poly_kernel(Xte, Xtr, best_d), Ktr );

    E = method_E_builder(methodID, best_p1, best_p2, W, B, Ktr, n);
    [U, lambda] = eig(E);

    % choose sorting order for method 7 (min eigen) otherwise descending
    if methodID == 7
        [~,I] = sort(diag(lambda),'ascend');
    else
        [~,I] = sort(diag(lambda),'descend');
    end
    U = U(:,I);

    p_use = min(best_p, size(U,2));
    Uproj = zeros(n, p_use);
    for ii=1:p_use
        u = real(U(:,ii));
        denom = sqrt(max(u'*Ktr*u, eps));
        Uproj(:,ii) = u ./ denom;
    end

    Ytrain = Ktr * Uproj;
    Ytest  = Kte * Uproj;

    mdl = fitcknn(Ytrain, ytr, 'NumNeighbors', 1);
    pred = predict(mdl, Ytest);

    accs(g) = sum(pred==yte)/numel(yte);
    all_preds = [all_preds; pred];
    all_labels = [all_labels; yte];
end

OA = nanmean(accs);
fprintf('Test OA (mean over folds) = %.4f (std=%.4f)\n', OA, nanstd(accs));

% compute confusion matrix over all folds if available
if ~isempty(all_labels)
    C = confusionmat(all_labels, all_preds);
    disp('Confusion matrix (aggregated):'); disp(C);
end

fprintf('Done.\n');
end

% ---------------------------
% helper: construct E matrix according to methodID
% ---------------------------
function E = method_E_builder(methodID, p1, p2, W, B, Ktr, n)
    switch methodID
        case 1
            % E = pinv(W*K_train_mean)*((1/n)*K_train_mean+B*K_train_mean);
            E = pinv_safe(W * Ktr) * ( (1/n) * Ktr + B * Ktr );
            E = p1 * E + p2 * E; % keep form but stage2 adjusts p1/p2 overall (we search p1/p2)
        case 2
            % E = (1/n)*K_train_mean + B*K_train_mean;
            E = (1/n) * Ktr + B * Ktr;
            E = p1 * E + p2 * E;
        case 3
            % E=pinv(W*K_train_mean)*(B*K_train_mean);
            E = pinv_safe(W * Ktr) * (B * Ktr);
            E = p1 * E;
        case 4
            % E=(1/n)*K_train_mean;
            E = (1/n)*Ktr;
            E = p1 * E;
        case 5
            % E=(1/n)*pinv(W*K_train_mean)*K_train_mean;
            E = (1/n) * pinv_safe(W * Ktr) * Ktr;
            E = p1 * E;
        case 6
            % E=B*K_train_mean;
            E = B * Ktr;
            E = p1 * E;
        case 7
            % E =W*K_train_mean;
            E = W * Ktr;
            E = p1 * E;
        otherwise
            error('Unknown methodID');
    end
end

% ---------------------------
% helper: safe pseudo-inverse multiplication
% ---------------------------
function Mpinv = pinv_safe(M)
    % stable pseudo-inverse for possibly rank-def matrices
    try
        Mpinv = pinv(M);
    catch
        Mpinv = pinv(M + 1e-8*eye(size(M)));
    end
end

% ---------------------------
% helper: center kernel (K -> Kc = K - 1_n K - K 1_n + 1_n K 1_n)
% ---------------------------
function Kc = center_kernel(K)
    n = size(K,1);
    one_n = ones(n)/n;
    Kc = K - one_n*K - K*one_n + one_n*K*one_n;
    % ensure symmetry
    Kc = (Kc + Kc')/2;
end

% ---------------------------
% helper: center cross-kernel K_x_tr using K_tr (centered)
% input K_xt: m x n, Ktr_cent: n x n (centered)
% we follow your earlier style: K_val_mean = K_val - mean(K_train,2)' - mean(K_val,2) + mean(K_train(:))
% ---------------------------
function Kx = center_kernel_cross(Kx_raw, Ktr_cent)
    % Kx_raw: m x n (val x train)
    % Ktr_cent: n x n (already centered)
    m = size(Kx_raw,1); n = size(Kx_raw,2);
    % compute means consistent with training centering
    mean_Ktr_rows = mean(Ktr_cent,2);     % n x 1
    mean_Kx_cols = mean(Kx_raw,2);        % m x 1
    grand = mean(Ktr_cent(:));
    Kx = Kx_raw - repmat(mean_Ktr_rows', m, 1) - repmat(mean_Kx_cols,1,n) + grand;
end
