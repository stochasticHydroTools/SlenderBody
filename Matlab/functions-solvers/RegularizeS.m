% Function to compute a regularized coordinate sbar from the fiber
% coordinates s over regularized length delta*L. 
% See Section 2.1 of Maxian et al. https://arxiv.org/pdf/2007.11728.pdf
% for details
function [sNew,regwt] = RegularizeS(s0,delta,L)
    sNew = s0;
    regwt = ones(length(s0),1);
    % Regularized version
    if (delta > 0 && delta < 0.5)
        x = 2*s0/L-1;
        regwt = tanh(((x+1)/delta))-tanh(((x-1)/delta))-1;
        sNew = s0;
        sNew(s0 < L/2) = regwt(s0 < L/2).*s0(s0 < L/2)+(1-regwt(s0 < L/2).^2).*delta*L/2;
        sNew(s0 > L/2) = L-flipud(sNew(s0 < L/2));
    elseif (delta >= 0.5)
        x = 2*s0/L-1;
        regwt = tanh((x+1)/0.5)-tanh((x-1)/0.5)-1;
        sNew = L/2*ones(length(s0),1);        
    end
end