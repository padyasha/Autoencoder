clear
close all
clc
addpath(genpath(strcat(pwd,'/drtoolbox')));

%% Data is loaded.

load data.mat
for i=1:30
    path=strcat(pwd,'\Training_A\malaria_',num2str(i),'.jpg');
    data_gray{i}=rgb2gray(imread(path));
    class{i}='Abnormal';
end 
for j=1:10
    path=strcat(pwd,'\Training_N\malaria_',num2str(j),'.jpg');
    data_gray{i+j}=rgb2gray(imread(path));
    class{i+j}='Normal';
end

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)

r=50;c=50;
final_data_re=zeros(r*c+1,size(class,2));
for i=1:size(class,2)
    if(strcmp(class{i},'Abnormal'))
        final_data_re(1,i)=0;
        im=imresize(data_gray{i},[r c]);
        final_data_re(2:end,i)=im(:);
    else
        final_data_re(1,i)=1;
        im=imresize(data_gray{i},[r c]);
        final_data_re(2:end,i)=im(:);
    end
end    

X=final_data_re(2:end,:);
X=X./max(max(X));
dataTrainG1 = X(:,grp2idx(class)==1)';
dataTrainG2 = X(:,grp2idx(class)==2)';

[mapped_data_G1, mapping_G1] = compute_mapping(dataTrainG1,'PCA', 10);
[mapped_data_G2, mapping_G2] = compute_mapping(dataTrainG2,'PCA', 10);
data_reduced=[mapped_data_G1;mapped_data_G2]';
% Use the SDAE to initialize a FFNN

 T=zeros(2,size(final_data_re,2));
 T(1,:)=final_data_re(1,:);
 T(2,:)=~T(1,:);
 data_reduced=X(1:500,:);
 Mdl=fitcsvm(data_reduced',class);
 Y=predict(Mdl,data_reduced');
 
% Train the FFNN
M=confusionmat(Y,class);
TT=[];
for i=1:size(Y,1)
    if(strcmp(Y{i},'normal'))
        TT=[TT;0 1];
    else
        TT=[TT;1 0];
    end
end
% Testing

plotroc(TT',T)
accuracy=(sum(sum(M.*eye(size(M))))/size(class,2))*100
