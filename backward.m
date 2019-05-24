%误差反向传播过程，求每层神经元误差以及权值和偏置更新 标准输出为label
%------------------------------output层-----------------------------------
%计算output层的神经元误差
tmp_delta_neuron_output=(tmp_neuron_output-label).*(1-tmp_neuron_output.^2);   %(1-y^2)*(y-label)
delta_neuron_ouput=reshape(tmp_delta_neuron_output,1,1,10);

%------------------------------c5层---------------------------------------
%计算c5层的神经元误差
tmp_delta_neuron_c5=tmp_delta_neuron_output'*wt_output';
tmp_delta_neuron_c5=tmp_delta_neuron_c5.*(1-tmp_neuron_c5.^2);  %导数*误差反传之和
delta_neuron_c5=reshape(tmp_delta_neuron_c5,1,1,120);
%计算output层权值更新与偏置
for j=1:num_map_output
    for i=1:num_map_c5
        delta_wt_output(i,j)=tmp_neuron_c5(i)*tmp_delta_neuron_output(j);
    end
    delta_bias_output(j)=tmp_delta_neuron_output(j);
end

%-----------------------------s4层----------------------------------------
%计算s4层的神经元误差
for i=1:num_map_s4
    %下面的循环是对s4每个map进行处理
    for x=1:h_s4
        for y=1:w_s4
            delta_neuron_s4(x,y,i)=reshape(wt_c5(x,y,i,:),1,120)*tmp_delta_neuron_c5'*(1-neuron_s4(x,y,i)^2);
        end
    end
end
%计算c5层的权值更新与偏置更新
for j=1:num_map_c5
    for i=1:num_map_s4
        %下面的循环对每个map处理
        for x=1:h_s4
            for y=1:w_s4
                delta_wt_c5(x,y,i,j)=neuron_s4(x,y,i)*delta_neuron_c5(:,:,j);
            end
        end
    end
    delta_bias_c5(j)=delta_neuron_c5(:,:,j);
end

%----------------------------c3层----------------------------------------
%计算c3层的神经元误差
for i=1:num_map_c3
    for x=1:h_c3
        for y=1:w_c3
            delta_neuron_c3(x,y,i)=wt_s4(i)*delta_neuron_s4(fix((x+1)/2),fix((y+1)/2),i)*(1-neuron_c3(x,y,i)^2)*(1/4);
        end
    end
end
%计算s4层的权值更新与偏置更新
delta_wt_s4=delta_wt_s4*0;
for i=1:num_map_s4
    for x=1:h_c3
        for y=1:w_c3
            delta_wt_s4(i)=delta_wt_s4(i)+neuron_c3(x,y,i)*delta_neuron_s4(fix((x+1)/2),fix((y+1)/2),i);
        end
    end
end
delta_wt_s4=delta_wt_s4/4;
delta_bias_s4=reshape(sum(sum(delta_neuron_s4)),1,16);

%---------------------------s2层---------------------------------------
%首先计算s2层的神经元误差，注意使用lechun的表格判断是否存在连接
delta_neuron_s2=delta_neuron_s2*0;
for i=1:num_map_s2
    for j=1:num_map_c3
        if tbl(i,j)==1 %判断两层的这两个map之间是否存在连接
            for x=1:h_c3
                for y=1:w_c3
                    delta_neuron_s2(x:x+4,y:y+4,i)=delta_neuron_s2(x:x+4,y:y+4,i)+delta_neuron_c3(x,y,j).*wt_c3(:,:,i,j);
                end
            end
        end
    end
end
delta_neuron_s2=delta_neuron_s2.*(1-neuron_s2.^2);

%计算权值更新与偏置更新值
for j=1:num_map_c3
    for i=1:num_map_s2
        %对一个卷积核的权值进行遍历
        for x=1:h_conv
            for y=1:w_conv
                delta_wt_c3(x,y,i,j)=sum(sum(delta_neuron_c3(:,:,j).*neuron_s2(x:x+h_c3-1,y:y+w_c3-1,i)));
            end
        end
    end
    delta_bias_c3(j)=sum(sum(delta_neuron_c3(:,:,j)));
end

%---------------------------c1层---------------------------------------
%计算c1层神经元误差
for i=1:num_map_c1
    for x=1:h_c1
        for y=1:w_c1
            delta_neuron_c1(x,y,i)=wt_s2(i)*delta_neuron_s2(fix((x+1)/2),fix((y+1)/2),i)*(1-neuron_c1(x,y,i)^2)*(1/4);
        end
    end
end
%计算s2层的权值更新与偏置更新
delta_wt_s2=delta_wt_s2*0;
for i=1:num_map_s2
    for x=1:h_c1
        for y=1:w_c1
            delta_wt_s2(i)=delta_wt_s2(i)+neuron_c1(x,y,i)*delta_neuron_s2(fix((x+1)/2),fix((y+1)/2),i);
        end
    end
end
delta_wt_s2=delta_wt_s2/4;
delta_bias_s2=reshape(sum(sum(delta_neuron_s2)),1,6);

%----------------------依据输入对c1更新-----------------------------
%对c1权值更新
for i=1:num_map_c1
    for x=1:h_conv
        for y=1:w_conv
            delta_wt_c1(x,y,i)=sum(sum(delta_neuron_c1(:,:,i).*img(x:x+h_c1-1,y:y+w_c1-1)));
        end
    end
    delta_bias_c1(i)=sum(sum(delta_neuron_c1(:,:,i)));
end



















