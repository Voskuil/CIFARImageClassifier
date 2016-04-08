function feat = extract_feature(image)
%% Description
% This function takes one image as input and returns HOG feature.
%
% Input: image
% Following VLFeat instruction, the input image should be SINGLE precision. 
% If not, the image is automatically converted to SINGLE precision.
%
% Output: feat
% The output is a vectorized HOG descriptor.
% The feature demension depends on the parameter, cellSize.
%
% VLFeat must be added to MATLAB search path. Please check the link below.
% http://www.vlfeat.org/install-matlab.html


%% check input data type
img=getimg(image);
if ~isa(img, 'single'), img = single(img); end;


%% extract HOG 
cellSize = 8;
hog = vl_hog(img, cellSize);
% imhog = vl_hog('render', hog, 'verbose');
% clf; imagesc(imhog); colormap gray;


%% feature - vectorized HOG descriptor
feat = hog(:);

end


function img = getimg(image)
    img=zeros(32,32,3);
    k=1;
    for n=1:3
        for i=1:32
            for j=1:32
                img(i,j,n)=image(k);
                k=k+1;
            end
        end
    end     
    img=uint8(img);
end

