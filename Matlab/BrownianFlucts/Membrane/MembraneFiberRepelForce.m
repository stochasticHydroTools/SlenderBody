function [FibForce,MemForce,Energy] = MembraneFiberRepelForce(Fib,Mem,Kster,...
    deltaP,Cutoffxy)
    % Upsample both the fiber and the membrane
    UpsampledMem =  [Mem.xgu(:) Mem.ygu(:) Mem.UpsamplingMatrix*Mem.h];
    UpsampledFib = Fib.Runi*reshape(Fib.Xt,3,[])';
    % For each fiber point, you know the x and y locations on the mem 
    % to look for
    NuM = length(Mem.xu);
    dxu = Mem.xu(2)-Mem.xu(1);
    nXYMax = 2*round(Cutoffxy/dxu); % Has to be even
    nPairs2 =0;
    Energy = 0;
    FibForce = zeros(Fib.Nx,3);
    MemForce = zeros(Mem.M^2,1);
    for iF=1:Fib.Nu
        pt = UpsampledFib(iF,:);
        RowFib = Fib.Runi(iF,:);
        flXY = floor(pt(1:2)/dxu);
        xPts = (-nXYMax/2+1:nXYMax/2)+flXY(1);
        yPts = (-nXYMax/2+1:nXYMax/2)+flXY(2);
        xPts = mod(xPts,NuM);
        yPts = mod(yPts,NuM);
        UpsampledInds = xPts*NuM + yPts'+1;
        UpsampledInds=UpsampledInds(:);
        % Look at the z locations
        MemLocs = UpsampledMem(UpsampledInds,:);
        rvecxy = pt(1:2) - MemLocs(:,1:2); 
        rvecxy = rvecxy - Mem.Lm*round(rvecxy/Mem.Lm);
        rxy=sqrt(sum(rvecxy.*rvecxy,2));
        rz = pt(3) - MemLocs(:,3);
        if (rxy<1e-12)
            keyboard
        end
        ThrowOut=rxy>Cutoffxy | rz < -deltaP;
        UpsampledInds(ThrowOut)=[];
        MemLocs(ThrowOut,:)=[];
        [fxy,fPrimexy]=StericRepXY(rxy,Cutoffxy/2);
        [fz,fPrimez]=StericRepZ(rz,deltaP);
        for iMP = 1:length(UpsampledInds)
            Energy = Energy+Kster*fz(iMP)*fxy(iMP)*Fib.wu(iF)*Mem.wtu;
            RowMem = Mem.UpsamplingMatrix(UpsampledInds(iMP),:);
            % z direction
            StericForceZ = Kster*fPrimez(iMP)*fxy(iMP)*Fib.wu(iF)*Mem.wtu;
            FibForce(:,3) = FibForce(:,3)-StericForceZ.*RowFib';
            MemForce = MemForce+StericForceZ*RowMem';
            % xy direction
            if (rxy(iMP) > 1e-10)
                StericForceXY = Kster*fPrimexy(iMP)*fz(iMP) ...
                    *rvecxy(iMP,1:2)/rxy(iMP)*Fib.wu(iF)*Mem.wtu;
            else
                % keyboard
                StericForceXY = [0 0]; % direction makes no sense
            end
            FibForce(:,1:2) = FibForce(:,1:2)-StericForceXY.*RowFib';
            nPairs2=nPairs2+1;
        end
    end
end

function [f,fPrime] = StericRepXY(r,delta)
    f = erfc(r/(delta*sqrt(2)));
    fPrime = -1/delta*exp(-r.^2/(2*delta.^2))*sqrt(2/pi);
end

function [f,fPrime]=StericRepZ(dh,delta)
    % This might need to be changed; we want to keep the fiber 
    % from going ABOVE the membrane. 
    %stForce = Kster*(1-dh/deltaP).*(dh<deltaP); % This is key; need to think about it
    f = (1+dh/delta).^2.*(dh>-delta);
    fPrime = 2/delta*(1+dh/delta).*(dh>-delta);
end

    