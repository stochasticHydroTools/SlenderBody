% G' and G'' from the stress tensor
function [Gp, Gdp] = getModuli(stresses,countf,dt,w,maxstr)
    stresses = stresses/maxstr;
    tarray=(0.5:countf)'*dt;
    Gp=2/(countf*dt)*sum(stresses.*sin(w*tarray)*dt);
    Gdp=2/(countf*dt)*sum(stresses.*cos(w*tarray)*dt);
end
    