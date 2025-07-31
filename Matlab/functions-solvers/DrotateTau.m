function JRot = DrotateTau(Xsin,Omega)
    % 3N x 3N matrix expressing derivative of rotation operation
    % JRot(i,j) = dXsNew_i / dOmega_j
    [N,~]=size(Xsin);
    JRot = zeros(3*N);
    for k=1:N
        Om1 = Omega(k,1);
        Om2 = Omega(k,2);
        Om3 = Omega(k,3);
        t1 = Xsin(k,1);
        t2 = Xsin(k,2);
        t3 = Xsin(k,3);
        nOm = norm(Omega(k,:));
        if (nOm > 1e-6)
        t5 = abs(Om1);
        t6 = abs(Om2);
        t7 = abs(Om3);
        t8 = sign(Om1);
        t9 = sign(Om2);
        t10 = sign(Om3);
        t11 = t5.^2;
        t12 = t6.^2;
        t13 = t7.^2;
        t14 = t11+t12+t13;
        t15 = 1.0./t14;
        t16 = sqrt(t14);
        t17 = cos(t16);
        t18 = sin(t16);
        t19 = 1.0./t16;
        t20 = t19.^3;
        t21 = t1.*t19;
        t22 = t2.*t19;
        t23 = t3.*t19;
        t33 = t17-1.0;
        t24 = Om1.*t21;
        t25 = Om1.*t22;
        t26 = Om2.*t21;
        t27 = Om1.*t23;
        t28 = Om2.*t22;
        t29 = Om3.*t21;
        t30 = Om2.*t23;
        t31 = Om3.*t22;
        t32 = Om3.*t23;
        t34 = -t21;
        t35 = -t22;
        t36 = -t23;
        t40 = Om1.*t1.*t5.*t8.*t20;
        t41 = Om1.*t1.*t6.*t9.*t20;
        t42 = Om2.*t2.*t5.*t8.*t20;
        t43 = Om1.*t1.*t7.*t10.*t20;
        t44 = Om2.*t2.*t6.*t9.*t20;
        t45 = Om3.*t3.*t5.*t8.*t20;
        t46 = Om2.*t2.*t7.*t10.*t20;
        t47 = Om3.*t3.*t6.*t9.*t20;
        t48 = Om3.*t3.*t7.*t10.*t20;
        t37 = -t26;
        t38 = -t29;
        t39 = -t31;
        t52 = t24+t28+t32;
        t53 = t34+t40+t42+t45;
        t54 = t35+t41+t44+t47;
        t55 = t36+t43+t46+t48;
        t49 = t25+t37;
        t50 = t27+t38;
        t51 = t30+t39;
        t56 = t19.*t33.*t52;
        t57 = -t56;
        mt1 = [t57-t18.*(Om2.*t3.*t5.*t8.*t20-Om3.*t2.*t5.*t8.*t20)+Om1.*t19.*t33.*t53+t5.*t8.*t18.*t34+t5.*t8.*t17.*t19.*t51+Om1.*t5.*t8.*t15.*t18.*t52+Om1.*t5.*t8.*t20.*t33.*t52,-t18.*(t23-Om1.*t3.*t5.*t8.*t20+Om3.*t1.*t5.*t8.*t20)+Om2.*t19.*t33.*t53+t5.*t8.*t18.*t35-t5.*t8.*t17.*t19.*t50+Om2.*t5.*t8.*t15.*t18.*t52+Om2.*t5.*t8.*t20.*t33.*t52,t18.*(t22-Om1.*t2.*t5.*t8.*t20+Om2.*t1.*t5.*t8.*t20)+Om3.*t19.*t33.*t53+t5.*t8.*t18.*t36+t5.*t8.*t17.*t19.*t49+Om3.*t5.*t8.*t15.*t18.*t52+Om3.*t5.*t8.*t20.*t33.*t52];
        mt2 = [t18.*(t23-Om2.*t3.*t6.*t9.*t20+Om3.*t2.*t6.*t9.*t20)+Om1.*t19.*t33.*t54+t6.*t9.*t18.*t34+t6.*t9.*t17.*t19.*t51+Om1.*t6.*t9.*t15.*t18.*t52+Om1.*t6.*t9.*t20.*t33.*t52,t57+t18.*(Om1.*t3.*t6.*t9.*t20-Om3.*t1.*t6.*t9.*t20)+Om2.*t19.*t33.*t54+t6.*t9.*t18.*t35-t6.*t9.*t17.*t19.*t50+Om2.*t6.*t9.*t15.*t18.*t52+Om2.*t6.*t9.*t20.*t33.*t52,-t18.*(t21+Om1.*t2.*t6.*t9.*t20-Om2.*t1.*t6.*t9.*t20)+Om3.*t19.*t33.*t54+t6.*t9.*t18.*t36+t6.*t9.*t17.*t19.*t49+Om3.*t6.*t9.*t15.*t18.*t52+Om3.*t6.*t9.*t20.*t33.*t52];
        mt3 = [-t18.*(t22+Om2.*t3.*t7.*t10.*t20-Om3.*t2.*t7.*t10.*t20)+Om1.*t19.*t33.*t55+t7.*t10.*t18.*t34+t7.*t10.*t17.*t19.*t51+Om1.*t7.*t10.*t15.*t18.*t52+Om1.*t7.*t10.*t20.*t33.*t52,t18.*(t21+Om1.*t3.*t7.*t10.*t20-Om3.*t1.*t7.*t10.*t20)+Om2.*t19.*t33.*t55+t7.*t10.*t18.*t35-t7.*t10.*t17.*t19.*t50+Om2.*t7.*t10.*t15.*t18.*t52+Om2.*t7.*t10.*t20.*t33.*t52,t57-t18.*(Om1.*t2.*t7.*t10.*t20-Om2.*t1.*t7.*t10.*t20)+Om3.*t19.*t33.*t55+t7.*t10.*t18.*t36+t7.*t10.*t17.*t19.*t49+Om3.*t7.*t10.*t15.*t18.*t52+Om3.*t7.*t10.*t20.*t33.*t52];
        Jac33 = reshape([mt1,mt2,mt3],3,3);
        else
        % Taylor series
        J11 = 1/6*(-3*Om1*t1 + 2*Om1*(Om3*t2 - Om2*t3) + ...
            3*(Om1*t1 + Om2*t2 + Om3*t3)); 
        J12 = 1/6*(-6*Om2*t1 + 3*Om1*t2 - (-6 + Om1^2 + Om2^2 + Om3^2)*t3 + ...
            2*Om2*(Om3*t2 - Om2*t3));
        J13 = 1/6*(-6*Om3*t1 + (-6 + Om1^2 + Om2^2 + Om3^2)*t2 + 3*Om1*t3 + ...
            2*Om3*(Om3*t2 - Om2*t3));
        J21 = 1/6*(3*Om2*t1 - 6*Om1*t2 + (-6 + Om1^2 + Om2^2 + Om3^2)*t3 + ...
            2*Om1*(-Om3*t1 + Om1*t3));
        J22 = 1/6*(-3*Om2*t2 + 2*Om2*(-Om3*t1 + Om1*t3) + 3*(Om1*t1 + ...
            Om2*t2 + Om3*t3));
        J23 = 1/6*(-((-6 + Om1^2 + Om2^2 + Om3^2)*t1) - 6*Om3*t2 + 3*Om2*t3 + ...
            2*Om3*(-Om3*t1 + Om1*t3));
        J31 = 1/6*(3*Om3*t1 - (-6 + Om1^2 + Om2^2 + Om3^2)*t2 + 2*Om1*...
            (Om2*t1 - Om1*t2) - 6*Om1*t3);
        J32 = 1/6*((-6 + Om1^2 + Om2^2 + Om3^2)*t1 + 3*Om3*t2 + ...
            2*Om2*(Om2*t1 - Om1*t2) - 6*Om2*t3);
        J33 = 1/6*(2*Om3*(Om2*t1 - Om1*t2) - 3*Om3*t3 + 3*(Om1*t1 + ...
            Om2*t2 + Om3*t3));
        Jac33 = [J11 J12 J13; J21 J22 J23; J31 J32 J33];
        end
        JRot(3*k-2:3*k,3*k-2:3*k)=Jac33;
    end
end