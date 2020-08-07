% close all
% clear
% 
% fx = @(t) t.^2 .* cos(2*t);
% fy = @(t) t.^2 .* sin(2*t);
% fz = @(t)  t;
% t = linspace(0,6*pi,400);
% 
% figure
% plot3(fx(t), fy(t), fz(t));
% grid on

xTicks = get(gca, 'XTick');
yTicks = get(gca, 'YTick');
zTicks = get(gca, 'ZTick');
hold on
% coordinate transform: x = Aq
A = [1 gn 0
    0 1 0
    0 0 1];
%A = eye(3);

% draw x grid lines
x = [xTicks
    xTicks
    xTicks];
y = repmat([min(yTicks); max(yTicks); max(yTicks) ], 1, length(xTicks));
z = repmat([min(zTicks); min(zTicks); max(zTicks) ], 1, length(xTicks));
X = A*[x(:)'; y(:)'; z(:)'];
line(reshape(X(1,:), 3, []),...
    reshape(X(2,:), 3, []),...
    reshape(X(3,:), 3, []), 'color', [.8 .8 .8]);

% draw y grid lines
y = [yTicks
    yTicks
    yTicks];
x = repmat([min(xTicks); max(xTicks); max(xTicks) ], 1, length(yTicks));
z = repmat([min(zTicks); min(zTicks); max(zTicks) ], 1, length(yTicks));
X = A*[x(:)'; y(:)'; z(:)'];
line(reshape(X(1,:), 3, []),...
    reshape(X(2,:), 3, []),...
    reshape(X(3,:), 3, []), 'color', [.8 .8 .8]);

% draw z grid lines
z = [zTicks
    zTicks
    zTicks];
x = repmat([min(xTicks); max(xTicks); max(xTicks) ], 1, length(zTicks));
y = repmat([max(yTicks); max(yTicks); min(yTicks) ], 1, length(zTicks));
X = A*[x(:)'; y(:)'; z(:)'];
line(reshape(X(1,:), 3, []),...
    reshape(X(2,:), 3, []),...
    reshape(X(3,:), 3, []), 'color', [.8 .8 .8]);

% draw grid planes
q{1} = [xTicks(1) xTicks(1) xTicks(end) xTicks(end)
    yTicks(1) yTicks(end) yTicks(end) yTicks(1)
    zTicks(1) zTicks(1) zTicks(1) zTicks(1)];
q{2} = [xTicks(end) xTicks(end) xTicks(end) xTicks(end)
    yTicks(1) yTicks(1) yTicks(end) yTicks(end)
    zTicks(1) zTicks(end) zTicks(end) zTicks(1)];
q{3} = [xTicks(1) xTicks(1) xTicks(end) xTicks(end)
    yTicks(end) yTicks(end) yTicks(end) yTicks(end)
    zTicks(1) zTicks(end) zTicks(end) zTicks(1)];
for i = 1:3
    x = A*q{i};
    fill3(x(1,:), x(2,:), x(3,:), [1 1 1]);
end

% cleanup and set view
axis off
view(-35, 30)