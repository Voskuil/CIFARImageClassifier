function [ kimgidx ] = Knn( k, data, img )
%KNN Summary of this function goes here
%   Detailed explanation goes here
    N=size(data,1);
    F=size(data,2);
    distance=sum((repmat(img,N,1)-data).^2,2);
    [dist,idx]=sort(distance,'ascend');
    kimgidx=idx(1:k);
end

