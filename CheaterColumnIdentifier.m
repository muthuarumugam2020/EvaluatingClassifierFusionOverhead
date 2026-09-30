% 1a. Load the Dataset and Extract X and Y
data = readtable('drebin215.csv');

% Extract the label column ('class') and create features table (X)
Y = categorical(data.class);
featuresTable = removevars(data, 'class');
featuresTable = featuresTable(:, vartype('numeric')); % Keep only numeric features
X = table2array(featuresTable);

% Train a single tree to see how it's cheating
debugTree = fitctree(X, Y);
imp = predictorImportance(debugTree);
[~, maxIdx] = max(imp);
cheaterColumnName = featuresTable.Properties.VariableNames{maxIdx};

fprintf('The suspicious column is: %s (Column Index: %d)\n', cheaterColumnName, maxIdx);