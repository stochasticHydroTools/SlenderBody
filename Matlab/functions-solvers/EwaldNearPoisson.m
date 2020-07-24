% Now add the near field
function nearphi = EwaldNearPoisson(pts,charges,xi,Lx,Ly,Lz,g)
    [nSource,~]=size(pts);
    nearphi=-2*xi/(4*pi*sqrt(pi))*charges;
    % Check that the Ewald parameter
    for iS=1:nSource
        for jS=1:nSource
            if (iS~=jS)
                % Find the nearest image p
                rvec=pts(iS,:)-pts(jS,:);
                rvec = calcShifted(rvec,g,Lx,Ly,Lz);
                r=norm(rvec);
                nearphi(iS)=nearphi(iS)+charges(jS)*erfc(xi*r)/(4*pi*r);
            end
        end
    end
end