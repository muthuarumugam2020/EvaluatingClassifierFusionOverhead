% 1. Load the Dataset
data = readtable('drebin215.csv');

% 2. Separate Labels (Y)
% Extract the label column explicitly by its name ('class')
Y = categorical(data.class); 

% 3. Separate Features (X)
% First, remove the 'class' column from the table so it doesn't get mixed in
featuresTable = removevars(data, 'class');

% Second, automatically drop any other text columns (like App Names or Hashes)
% This ensures ONLY numeric data (double) remains in the table
featuresTable = featuresTable(:, vartype('numeric'));

% Now it is 100% safe to convert to an array
X = table2array(featuresTable);

% 4. Apply the ReliefF Algorithm
fprintf('Running ReliefF Algorithm. This may take a few seconds...\n');
[rankedIndices, weights] = relieff(X, Y, 10);

% 5. Sort Weights for the Visual
sortedWeights = sort(weights, 'descend');

% 6. Generate Publication-Ready IEEE Figure with Enforced White Canvas
fig = figure('Color', 'w'); % Force white figure outer background
ax = gca;
hold(ax, 'on');

% Plot the bar chart (Professional dark blue bars)
bar(ax, sortedWeights, 'FaceColor', [0 0.27 0.54], 'EdgeColor', 'none'); 

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

% Dynamically lock x-axis to your exact number of features
xlim(ax, [0 size(X, 2) + 1]); 

% Add a solid black line at y=0 to clearly show where weights hit zero
yline(ax, 0, 'Color', 'k', 'LineWidth', 1);

% Add standard IEEE labels and title with enforced black text
xlabel('Features Ranked by Importance', 'FontWeight', 'bold', 'FontSize', 11, 'Color', 'k');
ylabel('Predictive Weight', 'FontWeight', 'bold', 'FontSize', 11, 'Color', 'k');
title('Fig. 1. ReliefF Feature Importance Distribution', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'k');

hold(ax, 'off');

% Automatically export a high-resolution 300 DPI image for IEEE
exportgraphics(fig, 'Figure1_ReliefF_Redundancy.png', 'Resolution', 300, 'BackgroundColor', 'white');