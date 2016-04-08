function [ Model] = train2(X,Y)
%TRAIN Summary of this function goes here
%   Detailed explanation goes here
    
    [training,centers] = extract_feature2(X,Y,false);
    xTrain = training.inputs;
    Y = training.classes;
    f=size(xTrain,2);
    w0= ones(f,1);
    nIter = 10000;
    W = [];
    classes = length(unique(Y));
    for i = 1:classes
        y_i = Y == (i-1);
        w_i = logisticRegressionWeights(xTrain,y_i,w0,nIter);
        W = [W, w_i];
    end
    Model=struct('W',W,'KCenters',centers);
end

function [w] = logisticRegressionWeights(xTrain, yTrain, w0, nIter)
   w = w0;
   %n = 10000;
   n = 0.5;
   for i = 1:nIter
      sigma = var(w);
      w = w + n*(yTrain'*xTrain-sigmoidProb(1,xTrain,w)'*xTrain)'; %- (1/(sigma+1))*n*w;
      %n = n * 0.999;
   end
end

function [p] = sigmoidProb(y, x, w)
    if y==0
        p = 1./(1+exp(x*w));
    else
        p = 1-1./(1+exp(x*w));
    end
end