% lit=1;
% e_wt_c1=e_wt_c1+norm(reshape(delta_wt_c1,1,get_num(delta_wt_c1)))^2/lit;
% e_wt_s2=e_wt_s2+norm(reshape(delta_wt_s2,1,get_num(delta_wt_s2)))^2/lit;
% e_wt_c3=e_wt_c3+norm(reshape(delta_wt_c3,1,get_num(delta_wt_c3)))^2/lit;
% e_wt_s4=e_wt_s4+norm(reshape(delta_wt_s4,1,get_num(delta_wt_s4)))^2/lit;
% e_wt_c5=e_wt_c5+norm(reshape(delta_wt_c5,1,get_num(delta_wt_c5)))^2/lit;
% e_wt_output=e_wt_output+norm(reshape(delta_wt_output,1,get_num(delta_wt_output)))^2/lit;
% 
% e_bias_c1=e_bias_c1+norm(reshape(delta_bias_c1,1,get_num(delta_bias_c1)))^2/lit;
% e_bias_s2=e_bias_s2+norm(reshape(delta_bias_s2,1,get_num(delta_bias_s2)))^2/lit;
% e_bias_c3=e_bias_c3+norm(reshape(delta_bias_c3,1,get_num(delta_bias_c3)))^2/lit;
% e_bias_s4=e_bias_s4+norm(reshape(delta_bias_s4,1,get_num(delta_bias_s4)))^2/lit;
% e_bias_c5=e_bias_c5+norm(reshape(delta_bias_c5,1,get_num(delta_bias_c5)))^2/lit;
% e_bias_output=e_bias_output+norm(reshape(delta_bias_output,1,get_num(delta_bias_output)))^2/lit;
% 
% wt_c1=wt_c1-delta_wt_c1*learning_rate/(sqrt(e_wt_c1)+eps);
% wt_s2=wt_s2-delta_wt_s2*learning_rate/(sqrt(e_wt_s2)+eps);
% wt_c3=wt_c3-delta_wt_c3*learning_rate/(sqrt(e_wt_c3)+eps);
% wt_s4=wt_s4-delta_wt_s4*learning_rate/(sqrt(e_wt_s4)+eps);
% wt_c5=wt_c5-delta_wt_c5*learning_rate/(sqrt(e_wt_c5)+eps);
% wt_output=wt_output-delta_wt_output*learning_rate/(sqrt(e_wt_output)+eps);
% 
% bias_c1=bias_c1-delta_bias_c1*learning_rate/(sqrt(e_bias_c1)+eps);
% bias_s2=bias_s2-delta_bias_s2*learning_rate/(sqrt(e_bias_s2)+eps);
% bias_c3=bias_c3-delta_bias_c3*learning_rate/(sqrt(e_bias_c3)+eps);
% bias_s4=bias_s4-delta_bias_s4*learning_rate/(sqrt(e_bias_s4)+eps);
% bias_c5=bias_c5-delta_bias_c5*learning_rate/(sqrt(e_bias_c5)+eps);
% bias_output=bias_output-delta_bias_output*learning_rate/(e_bias_output+eps);

wt_c1=wt_c1-delta_wt_c1*learning_rate;
wt_s2=wt_s2-delta_wt_s2*learning_rate;
wt_c3=wt_c3-delta_wt_c3*learning_rate;
wt_s4=wt_s4-delta_wt_s4*learning_rate;
wt_c5=wt_c5-delta_wt_c5*learning_rate;
wt_output=wt_output-delta_wt_output*learning_rate;

bias_c1=bias_c1-delta_bias_c1*learning_rate;
bias_s2=bias_s2-delta_bias_s2*learning_rate;
bias_c3=bias_c3-delta_bias_c3*learning_rate;
bias_s4=bias_s4-delta_bias_s4*learning_rate;
bias_c5=bias_c5-delta_bias_c5*learning_rate;
bias_output=bias_output-delta_bias_output*learning_rate;

