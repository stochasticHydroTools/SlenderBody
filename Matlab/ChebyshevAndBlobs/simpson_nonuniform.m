function result = simpson_nonuniform(x, f)
    %Simpson rule for irregularly spaced data.
    N = length(x) - 1;
    h = x(2:end)-x(1:end-1);
    result = 0.0;
    for i=2:2:N
        hph = h(i) + h(i - 1);
        result=result+ f(i,:) * ( h(i).^3 + h(i - 1).^3+ 3. * h(i) * h(i - 1) * hph )...
                     / ( 6 * h(i) * h(i - 1));
        result =result+ f(i - 1,:) * ( 2. * h(i - 1).^3 - h(i).^3+ 3. * h(i) * h(i - 1).^2)...
                     / ( 6 * h(i - 1) * hph);
        result =result+ f(i + 1,:) * ( 2. * h(i).^3 - h(i - 1).^3 + 3. * h(i - 1) * h(i).^2)...
                     / ( 6 * h(i) * hph );
    end
    if (mod((N + 1),2) == 0)
        result =result+ f(N+1,:) * ( 2 * h(N).^2 + 3. * h(N-1) * h(N))/ ( 6 *( h(N-1) + h(N)));
        result =result+ f(N,:)*(h(N).^2+ 3*h(N)* h(N-1))/( 6 * h(N-1));
        result =result- f(N-1,:)* h(N).^3/( 6 * h(N-1) * (h(N-1) + h(N)));
    end
end