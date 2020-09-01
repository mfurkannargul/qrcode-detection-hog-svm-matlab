clc
%% LOAD SVM CLASSIFIER TRAINED
load SVMModel

%% LOAD TEST IMAGES
folderTest = 'testImages';
imdsTest = imageDatastore(folderTest, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%% DISPLAY A SAMPLE TEST IMAGE
t = 1;
imageTest = readimage(imdsTest, t);
figure(3);
imshow(imageTest);
title('A Sample Test Image');
imdsTest.Labels(1)

%% MAKE PREDICTION BY USING SVM CLASSIFIER
numberTestImages = numel(imdsTest.Files)
start = 1;
correctPrediction = 0;
for i = start:numberTestImages
    imageTestUnresized = readimage(imdsTest,i);
    imageTest = imresize(imageTestUnresized,[128 64]);
    [featureVector,hogVisualization] = extractHOGFeatures(imageTest,'CellSize',cellSize);
    [prediction, scores] = predict(SVMModel,featureVector)
    figure(4);
    imshow(imageTest);
    title(strcat('Prediction:', string(prediction)))
    if (string(prediction) == string(imdsTest.Labels(start + i - 1)))
        correctPrediction = correctPrediction + 1;
    end
end

%% PREDICTION ACCURACY
accuracy = correctPrediction / (numberTestImages - start + 1) * 100
