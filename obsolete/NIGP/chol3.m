function R = chol3(X)

[N,jnk,E] = size(X);
R = zeros(N,N,E);

for i = 1:E
    R(:,:,i) = chol(X(:,:,i));
end
