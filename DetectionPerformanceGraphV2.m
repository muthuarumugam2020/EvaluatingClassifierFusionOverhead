% 1a. Load the Dataset and Extract X and Y
data = readtable('drebin215.csv');

% Extract the label column ('class') and create features table (X)
Y = categorical(data.class);
featuresTable = removevars(data, 'class');
% Add this right after removing 'class'
featuresTable = removevars(data, {'class', 'label'});

featuresTable = featuresTable(:, vartype('numeric')); % Keep only numeric features
X = table2array(featuresTable);


% Ensure X and Y are loaded and the 'class' text column is removed
% Y must be categorical. 
posClass = categories(Y);
% IMPORTANT: Assume the second category (alphabetically) is your 'Malware' class. 
% If your classes are 'B' and 'M', 'M' is posClass{2}. 
posTarget = posClass{2}; 

fprintf('Calculating ROC Curves and AUC (10-fold CV)...\n');

% 1. Create a fixed 10-fold cross-validation partition
cv = cvpartition(Y, 'KFold', 10);

% 2. Train Models and Extract Scores
% Decision Tree
dtModel = fitctree(X, Y, 'CVPartition', cv);
[pred_DT, score_DT] = kfoldPredict(dtModel);

% k-NN
knnModel = fitcknn(X, Y, 'CVPartition', cv);
[pred_KNN, score_KNN] = kfoldPredict(knnModel);

% SVM (Standardize helps the SVM generate stable distance scores)
svmModel = fitcsvm(X, Y, 'CVPartition', cv, 'Standardize', true);
[pred_SVM, score_SVM] = kfoldPredict(svmModel);

% 3. Calculate Fusion Score (Average of the votes)
% Convert predictions to logical (1 if Malware, 0 if Benign)
vote_DT = (pred_DT == posTarget);
vote_KNN = (pred_KNN == posTarget);
vote_SVM = (pred_SVM == posTarget);

% Fusion confidence score (0, 0.33, 0.66, or 1.0)
score_Fusion = (vote_DT + vote_KNN + vote_SVM) / 3;

% 4. Calculate ROC Coordinates and AUC
% We use the 2nd column of the score matrix, representing the positive class
[X_dt, Y_dt, ~, AUC_dt] = perfcurve(Y, score_DT(:,2), posTarget);
[X_knn, Y_knn, ~, AUC_knn] = perfcurve(Y, score_KNN(:,2), posTarget);
[X_svm, Y_svm, ~, AUC_svm] = perfcurve(Y, score_SVM(:,2), posTarget);
[X_fus, Y_fus, ~, AUC_fus] = perfcurve(Y, score_Fusion, posTarget);

% 5. Generate Publication-Ready IEEE Figure with Enforced White Canvas
fig = figure('Color', 'w'); % Force white figure outer background
ax = gca;
hold(ax, 'on');

% Plot curves with IEEE-compliant contrast and line styles
plot(X_dt, Y_dt, 'Color', [0.13 0.55 0.13], 'LineStyle', '-.', 'LineWidth', 1.6); % Forest green
plot(X_knn, Y_knn, 'Color', [0.00 0.45 0.74], 'LineStyle', ':',  'LineWidth', 1.8); % Blue
plot(X_svm, Y_svm, 'Color', [0.49 0.18 0.56], 'LineStyle', '--', 'LineWidth', 1.6); % Purple
plot(X_fus, Y_fus, 'Color', [0.85 0.10 0.10], 'LineStyle', '-',  'LineWidth', 2.0); % Red solid

% Enforce axes background, tick marks, and label styling
ax.Color = 'w';              % Force white plot area
ax.XColor = 'k';             % Black X-axis ticks and line
ax.YColor = 'k';             % Black Y-axis ticks and line
ax.FontSize = 10;
ax.FontName = 'Times New Roman'; % Standard IEEE font
ax.GridColor = [0.2 0.2 0.2]; % Crisp gray grid lines
ax.GridAlpha = 0.2;
grid(ax, 'on');
box(ax, 'on');

% Set axes limits
xlim(ax, [0 1]);
ylim(ax, [0 1]);

% Add standard IEEE labels and title
xlabel('False Positive Rate', 'FontWeight', 'bold', 'FontSize', 11, 'Color', 'k');
ylabel('True Positive Rate', 'FontWeight', 'bold', 'FontSize', 11, 'Color', 'k');
title('Fig. 2. ROC Curve Comparison of Classifiers', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');

% Format Legend for GitHub and Print Reproducibility
lgd = legend({sprintf('Decision Tree (AUC = %.4f)', AUC_dt), ...
              sprintf('k-NN (AUC = %.4f)', AUC_knn), ...
              sprintf('SVM (AUC = %.4f)', AUC_svm), ...
              sprintf('Multilevel Fusion (AUC = %.4f)', AUC_fus)}, ...
              'Location', 'southeast', 'FontSize', 9.5);

set(lgd, 'Color', 'w', 'EdgeColor', [0.5 0.5 0.5], 'TextColor', 'k');

hold(ax, 'off');

% Optional: Automatically export a high-resolution 300 DPI image for IEEE
exportgraphics(fig, 'Figure2_ROC_Curves.png', 'Resolution', 300, 'BackgroundColor', 'white');