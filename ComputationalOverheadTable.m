% 1a. Load the Dataset and Extract X and Y
data = readtable('drebin215.csv');

% Extract the label column ('class') and create features table (X)
Y = categorical(data.class);
featuresTable = removevars(data, 'class');
% Add this right after removing 'class'
featuresTable = removevars(data, {'class', 'label'});

featuresTable = featuresTable(:, vartype('numeric')); % Keep only numeric features
X = table2array(featuresTable);

% 1b. Create a fixed 10-fold cross-validation partition
% This ensures all models are evaluated on the exact same data splits
cv = cvpartition(Y, 'KFold', 10);

% 1. Create a single 80/20 train/test split for latency profiling
cvLatency = cvpartition(Y, 'HoldOut', 0.2);
idxTrain = training(cvLatency);
idxTest = test(cvLatency);

X_train = X(idxTrain, :); Y_train = Y(idxTrain, :);
X_test  = X(idxTest, :);  Y_test  = Y(idxTest, :);

fprintf('\nProfiling Computational Latency (Please wait)...\n');

% 2. Profile Decision Tree
tic; dtModel = fitctree(X_train, Y_train); timeTrainDT = toc;
tic; pred_DT = predict(dtModel, X_test); timePredDT = toc;

% 3. Profile k-Nearest Neighbors
tic; knnModel = fitcknn(X_train, Y_train); timeTrainKNN = toc;
tic; pred_KNN = predict(knnModel, X_test); timePredKNN = toc;

% 4. Profile Support Vector Machine
tic; svmModel = fitcsvm(X_train, Y_train); timeTrainSVM = toc;
tic; pred_SVM = predict(svmModel, X_test); timePredSVM = toc;

% 5. Profile Multilevel Fusion
% Fusion Training Time = Sum of training all base classifiers
timeTrainFusion = timeTrainDT + timeTrainKNN + timeTrainSVM;

% Fusion Prediction Time = Sum of base predictions + Majority Vote time
% Convert predictions to numeric for mode calculation
idx_DT = grp2idx(pred_DT);
idx_KNN = grp2idx(pred_KNN);
idx_SVM = grp2idx(pred_SVM);

tic; 
idx_Fusion = mode([idx_DT, idx_KNN, idx_SVM], 2);
timeVote = toc;

timePredFusion = timePredDT + timePredKNN + timePredSVM + timeVote;

% 6. Print Table II
fprintf('\n--- Table II: Computational Overhead Analysis (Seconds) ---\n');
fprintf('%-25s | %-16s | %-16s\n', 'Classifier', 'Training Latency', 'Prediction Time');
fprintf('------------------------------------------------------------------\n');
fprintf('%-25s | %-16.4f | %-16.4f\n', 'Decision Tree (DT)', timeTrainDT, timePredDT);
fprintf('%-25s | %-16.4f | %-16.4f\n', 'k-Nearest Neighbors', timeTrainKNN, timePredKNN);
fprintf('%-25s | %-16.4f | %-16.4f\n', 'Support Vector Machine', timeTrainSVM, timePredSVM);
fprintf('%-25s | %-16.4f | %-16.4f\n', 'Multilevel Fusion', timeTrainFusion, timePredFusion);
fprintf('------------------------------------------------------------------\n');