function [ Y] = classify2( Model,X)
%CLASSIFY Summary of this function goes here
%   Detailed explanation goes here
    W=Model.W;
    centers = Model.KCenters;
    Y=zeros(length(X(:,1)),1);
    [training,C] = extract_feature2(X,Y,centers);
    Xtest = training.inputs;
    [Y]=logisticRegressionClassify2(Xtest,W);
end

function [cls] = logisticRegressionClassify2(xTest, w)
   cls = zeros(length(xTest(:,1)),1);
   n=size(xTest,1);
   for j = 1:n
       classes = size(w,2);
       pList = [];
       for i = 1:classes
          p = sigmoidProb(1,xTest(j,:),w(:,i));
          pList = [pList,p];
       end
       [val,idx] = max(pList);
       cls(j) = idx-1;
   end
end



function [p] = sigmoidProb(y, x, w)
    if y==0
        p = 1./(1+exp(x*w));
    else
        p = 1-1./(1+exp(x*w));
    end
end