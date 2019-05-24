img=data_train(:,:,7);
figure(1)
imshow(img)
figure(2)
imshow(pooling(img,1))
figure(3)
imshow(max_pooling(img))