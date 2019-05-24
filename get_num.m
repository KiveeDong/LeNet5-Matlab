function ans=get_num(x)
    ans=1;
    dim=size(x);
    [a,len_dim]=size(dim);
    for i=1:len_dim
        ans=ans*dim(i);
    end
end