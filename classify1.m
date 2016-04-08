function [ Y ] = classify1( Model,X )
%CLASSIFY3 Summary of this function goes here
%   Detailed explanation goes here
    xTrain = Model.xTrain; 
    yTrain = Model.yTrain;
    N=size(X,1);
    K=100;
    KNN=zeros(N,K);
    xTest=zeros(N,496);
    for i=1:N
        xTest(i,:)=extract_feature(X(i,:));
    end
    X=xTest;
    for i=1:N
        KNN(i,:)=Knn(K,xTrain,X(i,:));
    end
    F=size(X,2);
    Y=zeros(N,1);
    for i=1:N
        yt=yTrain(KNN(i,:),:);
        xt=xTrain(KNN(i,:),:);
        Model=temp_train(xt,yt);
        p=Model.p;
        M=Model.M;
        V=Model.V;
        Y(i,1)=naiveBayesClassify(X(i,:),M,V,p);
    end
end

function [t] = naiveBayesClassify(xTest, M, V, p)
    numFeatures=size(xTest,2);
    numClass=size(V,2);
        maxp=-realmax;
        class=0;
        for a=1:numClass
            pf = log(normpdf(xTest(1,:),M(1:numFeatures,a)',sqrt(V(1:numFeatures,a)')));
            curp=logProd([pf,log(p(a))]);
            if maxp<exp(curp)
                maxp=exp(curp);
                class=a;
            end
        end
        t=class;
    t = int64(t)-1;
end


function [lp] = logProd(x)
	lp=sum(x);
end


function [ls] = logSum(x)
	p=max(x);
	ls=log(sum(exp(x-p)))+p;
end

