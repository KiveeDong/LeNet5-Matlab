function dst=pooling(src,weight)
    [sh,sw]=size(src);
    dh=sh/2;
    dw=sw/2;
    dst=zeros(dh,dw);
    for i=1:dh
        for j=1:dw
            dst(i,j)=weight*mean(mean(src(2*i-1:2*i,2*j-1:2*j)));
        end
    end
end