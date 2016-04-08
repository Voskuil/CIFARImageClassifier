function [ Y ] = classify( Model,X )
%CLASSIFY3 Summary of this function goes here
%   Detailed explanation goes here
    xTrain = Model.xTrain; 
    yTrain = Model.yTrain;
    N=size(X,1);
    K=5;
    KNN=zeros(N,K);
    xTest=zeros(N,496);
    for i=1:N
        xTest(i,:)=extract_feature(X(i,:));
    end
    X=xTest;
    for i=1:N
        KNN(i,:)=Knn(K,xTrain,xTest(i,:));
    end
    F=size(X,2);
    Y=zeros(N,1);
    for i=1:N
        yt=yTrain(KNN(i,:),:);
        Y(i,1)=mode(yt);
    end
end

