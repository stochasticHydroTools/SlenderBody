function [KT] = KT_mat(ds,U,V, clamp)
% Compute the K matrix

[m,n] = size(U);

for j = 1:n+1
    blocks{1,j} = speye(3);
end

Ind = 0;
if clamp
   Ind = 1;
end

for i= 2:(1+n)
    for j = 1:(1+n)
        if j < i
            blocks{i-Ind,j} = sparse(2,3);
        else
            blocks{i-Ind,j} = (-1)*ds*[V(:,i-1) -U(:,i-1)]'; %%% MODIFIED TO NEG
        end
    end
end

KT = cell2mat(blocks);

end