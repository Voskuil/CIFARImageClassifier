function [Model ] = train1( X,Y )
%TRAIN3 Summary of this function goes here
%   Detailed explanation goes here
    data=convertfeature(X);
    Model=struct('xTrain',data,'yTrain',Y);
end

function [ datafeatures ] = convertfeature( data )
%CONVERTFEATURE Summary of this function goes here
%   Detailed explanation goes here
    n=size(data,1);
    datafeatures=zeros(n,496);
    for i=1:n
        datafeatures(i,:)=extract_feature(data(i,:));
    end
end

