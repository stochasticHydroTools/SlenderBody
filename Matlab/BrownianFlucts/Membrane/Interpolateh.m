% This estimates the value of h(x,y) from two-dimensional 
% Fourier series. Options are: direct sum (UseNUFFT=0), 
% linear interpolation (UseNUFFT=-1), or NUFFT (=1)
function hInterp = Interpolateh(InterpPts,Mem,UseNUFFT)
    hmem=Mem.h;
    if (size(hmem,1)~=size(hmem,2))
        hmem=reshape(Mem.h,Mem.M,Mem.M);
    end
    hhat = fft2(hmem);
    % The direct calculation 
    if (UseNUFFT==0)
        nInterp = size(InterpPts,1);
        hInterp = zeros(nInterp,1);
        % Zero out unpaired mode
        for j = 1:size(InterpPts,1)
        hInterp(j) = 1/Mem.M^2*sum(sum(hhat.*exp(1i*Mem.kx*InterpPts(j,1)).*...
            exp(1i*Mem.ky*InterpPts(j,2))));
        end
        if (max(abs(imag(hInterp)))>1e-10)
            keyboard
        end
        hInterp=real(hInterp);
    elseif (UseNUFFT==-1)
        % Bilinear interpolation
        nInterp = size(InterpPts,1);
        hInterp = zeros(nInterp,1);
        UpsampMem = reshape(Mem.UpsamplingMatrix*Mem.h,Mem.Nu,Mem.Nu);
        % Zero out unpaired mode
        for j = 1:size(InterpPts,1)
            % Find closest x and y pts
            xpts = Mem.xu;
            dx = xpts(2)-xpts(1);
            InterpPts = InterpPts-floor(InterpPts/Mem.Lm)*Mem.Lm;
            x = InterpPts(j,1);
            flx = 1 + floor(x/dx);
            flxUp = flx+1;
            if (flxUp>length(xpts))
                flxUp=1;
            end
            x1 = Mem.xu(flx);
            x2 = Mem.xu(flx)+dx;
            y = InterpPts(j,2);
            fly = 1 + floor(y/dx);
            flyUp = fly+1;
            if (flyUp>length(xpts))
                flyUp=1;
            end
            y1 = Mem.xu(fly);
            y2 = Mem.xu(fly)+dx;
            w11 = (x2-x)*(y2-y)/dx^2;
            w12 = (x2-x)*(y-y1)/dx^2;
            w21 = (x-x1)*(y2-y)/dx^2;
            w22 = (x-x1)*(y-y1)/dx^2;
            hInterp(j) = w11*UpsampMem(fly,flx) + w12*UpsampMem(flyUp,flx) + ...
                w21*UpsampMem(fly,flxUp)+w22*UpsampMem(flyUp,flxUp);
        end
    else
        % Has to be done on the oversampled mesh
        dx=Mem.x(2)-Mem.x(1);
        gw=Mem.dx;
        N = length(Mem.x);
        nPair = (N-1)/2;
        L = N*dx;
        % Pad 
        hhatPad = zeros(Mem.Nu);
        hhatPad(1:nPair+1,1:nPair+1)=Mem.uRatio^2*hhat(1:nPair+1,1:nPair+1);
        hhatPad(1:nPair+1,end-nPair+1:end)=Mem.uRatio^2*hhat(1:nPair+1,end-nPair+1:end);
        hhatPad(end-nPair+1:end,1:nPair+1)=Mem.uRatio^2*hhat(end-nPair+1:end,1:nPair+1);
        hhatPad(end-nPair+1:end,end-nPair+1:end)=Mem.uRatio^2*hhat(end-nPair+1:end,end-nPair+1:end);
        ksqMult=Mem.ksqUp;
        ksqMult(abs(ksqMult)>2*(N/2*2*pi/L)^2)=0;
        hhatConv = hhatPad.*exp(gw^2*ksqMult/2);
        hConv = ifft2(hhatConv); % Upsampled grid
        hInterp = InterpFromGrid(Mem.xu,Mem.xu,hConv,InterpPts,gw);
    end
end
