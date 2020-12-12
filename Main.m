clc
clear
close all

training_data = readtable('plant_species_features.csv');
training_data = table2array(training_data);
training_data = training_data(2:end, 2:end);

rin = randperm(length(training_data));

plant_species_class_data = training_data(:, 1:end-1);
plant_species_class_data = plant_species_class_data(rin, :);

plant_species_condition_data = [training_data(:, 1:end-2), training_data(:, end)];
plant_species_condition_data = plant_species_condition_data(rin, :);

%%
%%______________________________Training_the_classifier_for_Classification_of_Plant_Species______________________________________

predictors = plant_species_class_data(:, 1:end-1);
response = plant_species_class_data(:,end);

template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 68 columns because this model was trained using 68 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);

pred_labels = partitionedModel.kfoldPredict;

figure;
plot(find(response==pred_labels), response(find(response==pred_labels)), 'g*');
xlabel('index');
ylabel('predicted class');
hold on;
plot(find(response~=pred_labels), response(find(response~=pred_labels)), 'rv');
legend('Correct Predictions', 'Wrong Predictions');
title('Plants Species Class Prediction');
set(gca,'xtick',(1:10),'yticklabel',{'Alstonia Scholaris', 'Arjun', 'Chinar', 'Gauva', ...
    'Jamun', 'Jetropha', 'Lemon', 'Mango', 'Pomegranate', 'Pongamia Pinnata'});
hold off;

ConfMat = ConfusionMatrix(pred_labels,response,10);
disp('Confusion Matrix for Plant Species Class Prediction');
disp(ConfMat);
acc = sum(diag(ConfMat))/sum(sum(ConfMat));
disp('Accuracy for Plant Species Class');
disp(acc);

figure;
cm = confusionchart(ConfMat, {'Alstonia Scholaris', 'Arjun', 'Chinar', 'Gauva', ...
    'Jamun', 'Jetropha', 'Lemon', 'Mango', 'Pomegranate', 'Pongamia Pinnata'});
cm.Title = 'Plant Species Classification using Quadratic SVM';

%%
%%______________________________Training_the_classifier_for_Classification_of_Plant's_Condition______________________________________

predictors = plant_species_condition_data(:, 1:end-1);
response = plant_species_condition_data(:,end);
response(response==2) = -1;

classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [-1; 1]);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 44 columns because this model was trained using 44 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 10);

pred_labels = partitionedModel.kfoldPredict;
response(response==-1) = 2;
pred_labels(pred_labels==-1) = 2;

figure;
plot(find(response==pred_labels), response(find(response==pred_labels)), 'g*');
xlabel('index');
ylabel('predicted class');
hold on;
plot(find(response~=pred_labels), response(find(response~=pred_labels)), 'rv');
legend('Correct Predictions', 'Wrong Predictions');
title('Plants Health Condtion Prediction');
set(gca,'xtick',[1,2],'yticklabel',{'Diseased', 'Healthy'});
hold off;

ConfMat = ConfusionMatrix(pred_labels,response,2);
disp('Confusion Matrix for Plant Species Class');
disp(ConfMat);
acc = sum(diag(ConfMat))/sum(sum(ConfMat));
disp('Accuracy for Plants Condition');
disp(acc);

figure;
cm = confusionchart(ConfMat, {'Diseased', 'Healthy'});
cm.Title = 'Plants Health Condition Classification using SVM with Gaussian Kernel';