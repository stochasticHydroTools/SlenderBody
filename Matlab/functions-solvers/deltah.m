% Smooth Gaussian delta function for cross linking
function val = deltah(a,N,L)
    if (N < 24)
        s = 0.1*L;
    elseif (N < 32)
       s = 0.07*L;
    else
       s = 0.05*L;
    end
    val = 1/sqrt(2*pi*s^2)*exp(-a.^2/(2*s^2));  % integrates to 1
end