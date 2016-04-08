function [training, centers] = extract_feature2(X,Y,train)
    data = X;
    classes = Y;
    n=size(data,1);
    inputs=zeros(128,186,n); %67712, 1058, 496, 8192, 512, 7936,8192
    for i=1:n
        img = reshape(data(i,:),32,32,3);
        [~, D1] = vl_phow(single(img),'color','rgb');
        feat = reshape(D1,128,186);
        inputs(:,:,i)=feat;
    end
    if train
        centers = train;
    else
        [centers] = vl_kmeans(inputs, 186); %100
    end
    numClusters = 186; %100
    H = zeros(n,numClusters);
    %batch = load('small_data_batch_1.mat');
    %data = batch.data;
    for j=1:n
        img = reshape(data(j,:),32,32,3);
        [~, D1] = vl_phow(single(img),'color','rgb');
        D = reshape(D1,128,186);
        D = double(D);
        for i=1:size(D,2)
            [~, k] = min(vl_alldist(D(:,i), centers)) ;
            H(j,k) = H(j,k) + 1;
        end
    end
    outputs = zeros(n,10);
    for i=1:n
        c = classes(i);
        outputs(i,c+1) = 1;
    end
    training = struct('inputs',H,'outputs',outputs,'classes',classes);
end