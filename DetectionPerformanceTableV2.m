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

% 2. Train Base Models with Cross-Validation
fprintf('Training Decision Tree (10-fold CV)...\n');
dtModel = fitctree(X, Y, 'CVPartition', cv);
pred_DT = kfoldPredict(dtModel);

fprintf('Training k-NN (10-fold CV)...\n');
knnModel = fitcknn(X, Y, 'CVPartition', cv);
pred_KNN = kfoldPredict(knnModel);

fprintf('Training SVM (10-fold CV)...\n');
svmModel = fitcsvm(X, Y, 'CVPartition', cv);
pred_SVM = kfoldPredict(svmModel);

% 3. Multilevel Fusion (Majority Vote)
% Convert categorical predictions to numeric indices (1 and 2) for math operations
idx_DT = grp2idx(pred_DT);
idx_KNN = grp2idx(pred_KNN);
idx_SVM = grp2idx(pred_SVM);
true_Y = grp2idx(Y); % Convert ground truth to numeric indices

% Find the most common prediction for each sample (Row-wise Mode)
idx_Fusion = mode([idx_DT, idx_KNN, idx_SVM], 2);

% 4. Calculate Metrics and Print Table
allPreds = {idx_DT, idx_KNN, idx_SVM, idx_Fusion};
modelNames = {'Decision Tree (DT)', 'k-Nearest Neighbors', 'Support Vector Machine', 'Multilevel Fusion'};

fprintf('\n--- Table I: Classification Performance on Drebin-215 (%%) ---\n');
fprintf('%-25s | %-8s | %-9s | %-8s | %-8s\n', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-Score');
fprintf('------------------------------------------------------------------------\n');

for i = 1:4
    preds = allPreds{i};
    
    % Generate Confusion Matrix
    cm = confusionmat(true_Y, preds);
    
    % Extract True Positives, True Negatives, False Positives, False Negatives
    % (Assuming class 2 is the 'Malware' positive class)
    TN = cm(1,1); FP = cm(1,2);
    FN = cm(2,1); TP = cm(2,2);
    
    % Calculate Standard Metrics
    Accuracy  = (TP + TN) / sum(cm(:)) * 100;
    Precision = TP / (TP + FP) * 100;
    Recall    = TP / (TP + FN) * 100;
    F1_Score  = 2 * (Precision * Recall) / (Precision + Recall);
    
    % Print row for the table
    fprintf('%-25s | %-8.2f | %-9.2f | %-8.2f | %-8.2f\n', modelNames{i}, Accuracy, Precision, Recall, F1_Score);
end
fprintf('------------------------------------------------------------------------\n');