function [ Y] = classifySVM( Model,X)
    fins = [];
    n=size(X,1);
    xTrain=Model.xTrain;
%     xTest=zeros(n,496);
%     for i=1:n
%         xTest(i,:)=extract_feature(X(i,:));
%     end
    X2=double(X);
    W = Model.W;
    w0 = Model.Bias;
    for i = 1:10
        w = W(i,:);
        idx = w0(i,2);
        b0 = w0(i,1);
        b = b0 - 0.64* w * xTrain(idx,:)'; %* 0.8
        fin = w * X2' + b;
        fins = [fin', fins];
    end
    [maxA,ind] = max(fins');
    Y = 10-ind;
    Y = Y';
end