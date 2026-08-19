function M = RPYMatrixWithWall(r_vectors,eta, a)
  %
  % Returns the mobility at the blob level in free space
  %  
  %%%
  % Variables
  if (size(r_vectors,2)~=3)
      r_vectors=reshape(r_vectors,3,[])';
  end
  N = size(r_vectors,1);
  inva = 1.0 / a;
  norm_fact_f = 1.0 / (8.0 * pi * eta * a);
    
  rx_vec = r_vectors(:,1);
  ry_vec = r_vectors(:,2);
  rz_vec = r_vectors(:,3);
  M = zeros(3*N);
    
  % Loop over image boxes and then over particles
  for i =1:N
    rxi = rx_vec(i);
    ryi = ry_vec(i);
    rzi = rz_vec(i);
          for j=i:N
            % Compute vector between particles i and j
            rx = rxi - rx_vec(j);
            ry = ryi - ry_vec(j);
            rz = rzi - rz_vec(j);      
               
            % 1. Compute mobility for pair i-j, if i==j use self-interation
            rx = rx * inva;
            ry = ry * inva;
            rz = rz * inva;
            if i == j
              Mxx = 4/3;
              Mxy = 0;
              Mxz = 0;
              Myy = Mxx;
              Myz = 0;
              Mzz = Mxx;          
            else
              % Normalize distance with hydrodynamic radius
              r2 = rx*rx + ry*ry + rz*rz;
              r = sqrt(r2);
              
              invr = 1.0 / r;
              invr2 = invr * invr;

              if r > 2
                c1 = 1.0 + 2.0 / (3.0 * r2);
                c2 = (1.0 - 2.0 * invr2) * invr2;
                Mxx = (c1 + c2*rx*rx) * invr;
                Mxy = (     c2*rx*ry) * invr;
                Mxz = (     c2*rx*rz) * invr;
                Myy = (c1 + c2*ry*ry) * invr;
                Myz = (     c2*ry*rz) * invr;
                Mzz = (c1 + c2*rz*rz) * invr;
              else
                c1 = 4/3 * (1 - 9/32 * r);
                c2 = 4/3 * 3/32 * invr;
                Mxx = c1 + c2 * rx*rx ;
                Mxy =      c2 * rx*ry ;
                Mxz =      c2 * rx*rz ;
                Myy = c1 + c2 * ry*ry ;
                Myz =      c2 * ry*rz ;
                Mzz = c1 + c2 * rz*rz ;
              end
            end
            Myx = Mxy;
            Mzx = Mxz;
            Mzy = Myz;

            % Wall correction
            rz = (rzi + rz_vec(j)) * inva;
            hj = rz_vec(j) * inva;

            if i == j
              invZi = 1.0 / hj;
              invZi3 = invZi * invZi * invZi;
              invZi5 = invZi3 * invZi * invZi;
            
              Mxx = Mxx-(9.0 * invZi - 2.0 * invZi3 + invZi5 ) / 12.0;
              Myy = Myy-(9.0 * invZi - 2.0 * invZi3 + invZi5 ) / 12.0;
              Mzz = Mzz-(9.0 * invZi - 4.0 * invZi3 + invZi5 ) / 6.0 ;  
            else
              h_hat = hj / rz;
              invR = 1.0 / sqrt(rx*rx + ry*ry + rz*rz);
              ex = rx * invR;
              ey = ry * invR;
              ez = rz * invR;
              invR3 = invR * invR * invR;
              invR5 = invR3 * invR * invR;
                  
              fact1 = -(3.0*(1.0+2.0*h_hat*(1.0-h_hat)*ez*ez) * invR + 2.0*(1.0-3.0*ez*ez) * invR3 - 2.0*(1.0-5.0*ez*ez) * invR5)  / 3.0;
              fact2 = -(3.0*(1.0-6.0*h_hat*(1.0-h_hat)*ez*ez) * invR - 6.0*(1.0-5.0*ez*ez) * invR3 + 10.0*(1.0-7.0*ez*ez) * invR5) / 3.0;
              fact3 =  ez * (3.0*h_hat*(1.0-6.0*(1.0-h_hat)*ez*ez) * invR - 6.0*(1.0-5.0*ez*ez) * invR3 + 10.0*(2.0-7.0*ez*ez) * invR5) * 2.0 / 3.0;
              fact4 =  ez * (3.0*h_hat*invR - 10.0*invR5) * 2.0 / 3.0;
              fact5 = -(3.0*h_hat*h_hat*ez*ez*invR + 3.0*ez*ez*invR3 + (2.0-15.0*ez*ez)*invR5) * 4.0 / 3.0;
    
              Mxx = Mxx+fact1 + fact2 * ex*ex;
              Mxy = Mxy+fact2 * ex*ey;
              Mxz = Mxz+fact2 * ex*ez + fact3 * ex;
              Myx = Myx+fact2 * ey*ex;
              Myy = Myy+fact1 + fact2 * ey*ey;
              Myz = Myz+fact2 * ey*ez + fact3 * ey;
              Mzx = Mzx+fact2 * ez*ex + fact4 * ex;
              Mzy = Mzy+fact2 * ez*ey + fact4 * ey;
              Mzz = Mzz+fact1 + fact2 * ez*ez + fact3 * ez + fact4 * ez + fact5;
            end
        M(3*i-2:3*i,3*j-2:3*j)=[Mxx Mxy Mxz; Myx Myy Myz; Mzx Mzy Mzz]* norm_fact_f;
        M(3*j-2:3*j,3*i-2:3*i)=[Mxx Mxy Mxz; Myx Myy Myz; Mzx Mzy Mzz]'* norm_fact_f;
          end
  end
end

