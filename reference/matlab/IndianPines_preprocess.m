% IndianPines_preprocess.m
% expects data.mat (N x D), label.mat (N x 1)
clear; clc; close all;
load('data.mat'); load('label.mat'); % data, label
numGroups = 10;
classes = unique(label);
for g=1:numGroups
    rng(2025+g);
    traindata=[]; trainlabels=[]; valdata=[]; vallabels=[]; testdata=[]; testlabels=[];
    for i=1:numel(classes)
        idx = find(label==classes(i));
        N = numel(idx);
        idx = idx(randperm(N));
        train_num = max(1, round(0.01*N));
        val_num = max(1, round(0.05*N));
        if train_num + val_num >= N
            val_num = max(1, N - train_num - 1);
        end
        train_idx = idx(1:train_num);
        val_idx = idx(train_num+1:train_num+val_num);
        test_idx = idx(train_num+val_num+1:end);
        traindata = [traindata; data(train_idx,:)];
        trainlabels = [trainlabels; label(train_idx)];
        valdata = [valdata; data(val_idx,:)];
        vallabels = [vallabels; label(val_idx)];
        testdata = [testdata; data(test_idx,:)];
        testlabels = [testlabels; label(test_idx)];
    end
    save(sprintf('traindata%02d.mat',g),'traindata');
    save(sprintf('trainlabels%02d.mat',g),'trainlabels');
    save(sprintf('valdata%02d.mat',g),'valdata');
    save(sprintf('vallabels%02d.mat',g),'vallabels');
    save(sprintf('testdata%02d.mat',g),'testdata');
    save(sprintf('testlabels%02d.mat',g),'testlabels');
end
fprintf('Indian Pines splits saved.\n');
