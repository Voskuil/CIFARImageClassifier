function [ Model ] = temp_train( X,Y )
%TRAIN Summary of this function goes here
%   Detailed explanation goes here
      n=size(X,1);
      f=size(X,2);
%       xTrain=zeros(n,496);
%       for i=1:n
%           xTrain(i,:)=extract_feature(X(i,:));
%       end
    Y=double(Y)+double(1);
    xTrain=double(X);
    [M,V] = likelihood(xTrain,Y);
    p = prior(Y);
    Model=struct('M',M,'V',V,'p',p);
    
end

function [lp] = logProd(x)
	lp=sum(x);
end


function [ls] = logSum(x)
	p=max(x);
	ls=log(sum(exp(x-p)))+p;
end



function [M, V] = likelihood(xTrain, yTrain)
    len=length(yTrain);
    numFeatures=size(xTrain,2);
    mm=max(yTrain);
    M=zeros(numFeatures,mm);
    V=zeros(numFeatures,mm);
    S=zeros(1,mm);
    for n=1:len
         S(yTrain(n))=S(yTrain(n))+1;
         M(1:numFeatures,yTrain(n))=M(1:numFeatures,yTrain(n))+xTrain(n,1:numFeatures)';
    end
    M=bsxfun(@rdivide,M,S);
    for n=1:len
        V(1:numFeatures,yTrain(n))=V(1:numFeatures,yTrain(n))+(xTrain(n,1:numFeatures)'- M(1:numFeatures,yTrain(n))).^2;
    end
    V=bsxfun(@rdivide,V,S);
end


function [p] = prior(yTrain)
	len=length(yTrain);
	mm=max(yTrain);
	S=zeros(mm,1);
	for n=1:len
		S(yTrain(n))=S(yTrain(n))+1;
	end
	p=zeros(mm,1);
	for n=1:mm
		p(n)=S(n)/len;
	end
end



