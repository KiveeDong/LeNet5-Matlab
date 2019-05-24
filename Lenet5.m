clc
clear
train_num=10000;
test_num=2000;
data_train=zeros(32,32,60000)-1;
data_train(3:30,3:30,:)=loadMNISTImages('train-images.idx3-ubyte');
label_train=loadMNISTLabels('train-labels.idx1-ubyte');
stdout_train=-ones(10,60000)*0.8;
for i=1:60000
    stdout_train(label_train(i)+1,i)=0.8;
end

data_test=zeros(32,32,10000)-1;
data_test(3:30,3:30,:)=loadMNISTImages('t10k-images.idx3-ubyte');
label_test=loadMNISTLabels('t10k-labels.idx1-ubyte');
stdout_test=-ones(10,10000)*0.8;
for i=1:10000
    stdout_test(label_test(i)+1,i)=0.8;
end

%---------------------首先定义网络参数-------------------------------------
%1.各层map(通道)数目
num_map_input=1;
num_map_c1=6;
num_map_s2=6;
num_map_c3=16;
num_map_s4=16;
num_map_c5=120;
num_map_output=10;

%2.各层操作的图像高与宽以及卷积核尺寸
h_input=32;w_input=32;
h_c1=28;w_c1=28;
h_s2=14;w_s2=14;
h_c3=10;w_c3=10;
h_s4=5 ;w_s4=5;
h_c5=1 ;w_c5=1;
h_output=1;w_output=1;
h_conv=5   ;w_conv=5;
h_pooling=2;w_pooling=2;

%3.各层神经元，权值，偏置数目
num_neuron_input=32*32;
num_neuron_c1=6*28*28;
num_neuron_s2=6*14*14;
num_neuron_c3=16*10*10;
num_neuron_s4=5*5*16;
num_neuron_c5=1*120;
num_neuron_output=10;

num_weight_c1=5*5*6;
num_weight_s2=6;
num_weight_c3=5*5*16*6;
num_weight_s4=16;
num_weight_c5=5*5*16*120;
num_weight_output=120*10;

num_bias_c1=6;
num_bias_s2=6;
num_bias_c3=16;
num_bias_s4=16;
num_bias_c5=120;
num_bias_output=10;

%4.训练相关的参数
maxepochs=100;
accuracy_req=0.985;
learning_rate=0.01;
eps=1e-8;
e_wt_c1=0;e_bias_c1=0;
e_wt_s2=0;e_bias_s2=0;
e_wt_c3=0;e_bias_c3=0;
e_wt_s4=0;e_bias_s4=0;
e_wt_c5=0;e_bias_c5=0;
e_wt_output=0;e_bias_output=0;
%---------------------------参数定义完毕------------------------------------



%-----------------------------网络变量的定义--------------------------------
wt_c1=(rand(1,num_weight_c1)-0.5)*(sqrt(6/175));wt_c1=reshape(wt_c1,5,5,6);
wt_s2=(rand(1,num_weight_s2)-0.5)*(sqrt(6/5));
wt_c3=(rand(1,num_weight_c3)-0.5)*(sqrt(6/550));wt_c3=reshape(wt_c3,5,5,6,16);
wt_s4=(rand(1,num_weight_s4)-0.5)*(sqrt(6/5));
wt_c5=(rand(1,num_weight_c5)-0.5)*(sqrt(6/3400));wt_c5=reshape(wt_c5,5,5,16,120);
wt_output=(rand(1,num_weight_output)-0.5)*(sqrt(6/130));wt_output=reshape(wt_output,120,10);
bias_c1=zeros(1,num_bias_c1);
bias_s2=zeros(1,num_bias_s2);
bias_c3=zeros(1,num_bias_c3);
bias_s4=zeros(1,num_bias_s4);
bias_c5=zeros(1,num_bias_c5);
bias_output=zeros(1,num_bias_output);

delta_wt_c1=zeros(1,num_weight_c1);delta_wt_c1=reshape(delta_wt_c1,5,5,6);%权值及偏置更新量
delta_wt_s2=zeros(1,num_weight_s2);
delta_wt_c3=zeros(1,num_weight_c3);delta_wt_c3=reshape(delta_wt_c3,5,5,6,16);
delta_wt_s4=zeros(1,num_weight_s4);
delta_wt_c5=zeros(1,num_weight_c5);delta_wt_c5=reshape(delta_wt_c5,5,5,16,120);
delta_wt_output=zeros(1,num_weight_output);delta_wt_output=reshape(delta_wt_output,120,10);
delta_bias_c1=zeros(1,num_bias_c1);
delta_bias_s2=zeros(1,num_bias_s2);
delta_bias_c3=zeros(1,num_bias_c3);
delta_bias_s4=zeros(1,num_bias_s4);
delta_bias_c5=zeros(1,num_bias_c5);
delta_bias_output=zeros(1,num_bias_output);

%Lechun table
% tbl=[0,1,1,1,0,0,0,1,1,0,0,0,0,1,0,0
%      0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,0
%      0,0,0,1,1,1,0,0,0,1,1,0,1,0,0,0
%      1,0,0,0,1,1,0,0,0,0,1,1,0,1,0,0
%      1,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0
%      1,1,1,0,0,0,1,1,0,0,0,0,1,0,0,0];
 tbl=[1,0,0,0,1,1,1,0,0,1,1,1,1,0,1,1
      1,1,0,0,0,1,1,1,0,0,1,1,1,1,0,1
      1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1
      0,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1
      0,0,1,1,1,0,0,1,1,1,1,0,1,1,0,1
      0,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1];

%每层的输出及神经元误差
%neuron_input=zeros(h_input,w_input,num_map_input);%每层输出
neuron_c1=zeros(h_c1,w_c1,num_map_c1);
neuron_s2=zeros(h_s2,w_s2,num_map_s2);
neuron_c3=zeros(h_c3,w_c3,num_map_c3);
neuron_s4=zeros(h_s4,w_s4,num_map_s4);
neuron_c5=zeros(h_c5,w_c5,num_map_c5);
neuron_output=zeros(h_output,w_output,num_map_output);

%delta_neuron_input=zeros(h_input,w_input,num_map_input);%每层神经元误差，但输入层的好像没用
delta_neuron_c1=zeros(h_c1,w_c1,num_map_c1);
delta_neuron_s2=zeros(h_s2,w_s2,num_map_s2);
delta_neuron_c3=zeros(h_c3,w_c3,num_map_c3);
delta_neuron_s4=zeros(h_s4,w_s4,num_map_s4);
delta_neuron_c5=zeros(h_c5,w_c5,num_map_c5);
delta_neuron_ouput=zeros(h_output,w_output,num_map_output);
%---------------------------变量定义完毕-----------------------------------


%--------------------------训练过程---------------------------------------
for epoch=1:maxepochs
    for img_ids=1:100:train_num
    %for img_id=(epoch-1)*600+1:epoch*600
    fprintf('第%d轮,正在训练第%d--%d幅图片...\n',epoch,img_ids,img_ids+99);
        for img_id=img_ids:img_ids+99
            %fprintf('正在训练第%d-%d幅图片...\n',epoch,img_id);
            img=data_train(:,:,img_id);
            label=stdout_train(:,img_id);
            forward;
            backward;
            update;
        end
    end    
    test;   
end
%-------------------------训练结束-----------------------------------------




