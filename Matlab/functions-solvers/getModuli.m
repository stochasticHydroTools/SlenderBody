% G' and G'' from the stress tensor
function [Gp, Gdp,er] = getModuli(stresses,Tf,tarray,w,maxstr)
    stresses = stresses/maxstr;
    dt = tarray(2)-tarray(1);
    Gp=2/(Tf)*sum(stresses.*sin(w*tarray)*dt);
    Gdp=2/(Tf)*sum(stresses.*cos(w*tarray)*dt);
    er=stresses-(Gp*sin(tarray*w)+Gdp*cos(tarray*w));
end
    