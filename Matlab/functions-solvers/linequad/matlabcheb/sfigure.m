function h = sfigure(f)
if nargin==0
    h = figure();
elseif ishandle(f)
    set(0, 'CurrentFigure', f);
else
    h = figure(f);
end
