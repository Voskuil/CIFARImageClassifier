function [ Model] = trainSVM(X,Y)
    n=size(X,1);
    f=size(X,2);
%     xTrain=zeros(n,496);
%     for i=1:n
%         xTrain(i,:)=extract_feature(X(i,:));
%     end
    X=double(X);
    W=[];
    alphas=[];
    W0 = [];
    for j = 1:10
        labels1 = Y == (j-1);
        labels2 = labels1 - 1;
        y = labels1 + labels2;
        y=double(y);
        ell=size(X,1);
        H=(X*X').*(y*y');
        f=-ones(ell,1);
        A=-eye(ell);
        a = zeros(ell,1);
        B = [y';zeros(ell-1,ell)];
        b = zeros(ell,1);
        %alpha=quadprog(H+eye(ell)*0.001,f,A,a,B,b);
        alpha=qp([],H+eye(ell)*0.001,f);
        w=(alpha.*y)'*X;
        alphas = [alphas, alpha];
        W = [W; w];
        noAlpha = true;
        idx = 1;
        while noAlpha
            if alpha(idx) > 0.001
                noAlpha = false;
            elseif idx == n
                noAlpha = false;
            end
            idx = idx + 1;
        end
        idx = idx-1;
        w0 = 1/y(idx);
        W0 = [W0;[w0,idx]];
    end
    Model=struct('W',W,'Alphas',alphas,'Bias',W0,'xTrain',X);
end

    
