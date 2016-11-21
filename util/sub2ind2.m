function idx = sub2ind2(D,i,j)
% Get linear indexes from all combinations of two subscripts
% D = #rows, i = row subscript, j = column subscript
i = i(:); j = j(:)';
idx =  reshape(bsxfun(@plus,D*(j-1),i),1,[]);

