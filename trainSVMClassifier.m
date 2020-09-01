%% LOAD TRAINING IMAGES
clc, clear
folder = 'trainImages';
imdsTrain = imageDatastore(folder, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
numberImages = numel(imdsTrain.Files)

%% EXTRACT A SAMPLE HOG FEATURE VECTOR
k = 5;
img_unresized = readimage(imdsTrain,k);
img = imresize(img_unresized,[128 64]);
%img = img_unresized;
imshow(img);
[featureVector,hogVisualization] = extractHOGFeatures(img,'CellSize',[2 2]);
figure(1);
imshow(img);
hold on;
plot(hogVisualization)
title('hogVisualization')

%% DETERMINE CELL SIZE AND FEATURE SIZE
cellSize = [2 2]
hogFeatureSize = length(featureVector)

%% EXTRACT HOG FEATURE VECTORS OF TRAIN SET
arrayTrainingFeatures = zeros(numberImages,hogFeatureSize,'single');
for k = 1:numberImages
    img_unresized = readimage(imdsTrain,k);
    img = imresize(img_unresized,[128 64]);
    %img = img_unresized;
    size(img)
    [featureVector,hogVisualization] = extractHOGFeatures(img,'CellSize', cellSize);
    size(featureVector)
    for j = 1:length(hogFeatureSize)
        arrayTrainingFeatures(k,j) = featureVector(1,j);
    end
end
trainingLabels = imdsTrain.Labels;
size(trainingLabels)

%% TRAIN SVM CLASSIFIER
SVMModel = fitcsvm(arrayTrainingFeatures, trainingLabels)

%% TEST ON TRAINING IMAGES
start = 1;
correctPrediction = 0;
for i = start:numberImages
    image_unresized = readimage(imdsTrain,i);
    image = imresize(image_unresized,[128 64]);
    %image = imgage_unresized;
    [featureVector,hogVisualization] = extractHOGFeatures(image,'CellSize',cellSize);
    [prediction, scores] = predict(SVMModel,featureVector)
    figure(2);
    imshow(image);
    title(strcat('Prediction:', string(prediction)))
    if (string(prediction) == string(trainingLabels(start + i - 1)))
        correctPrediction = correctPrediction + 1;
    end
end

%% PREDICTION ACCURACY
accuracy = correctPrediction / (numberImages - start) * 100

%% SAVE THE TRAINED CLASSIFIER FOR FURTHER USE
%save SVMModel
save('SVMModel', '-v7.3')