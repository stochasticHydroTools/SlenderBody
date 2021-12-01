function rotated_x=rotate(x,Omega)
    nOm = norm(Omega);
    Omhat = Omega/norm(Omega);
    Px = Omhat*dot(Omhat,x);
    rotated_x = Px+cos(nOm)*(x-Px)+sin(nOm)*cross(Omhat,x);
    if (nOm < 1e-10)
        rotated_x = x;
    end
end
