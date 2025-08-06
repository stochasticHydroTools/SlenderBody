function PlotFil(Disc)
    Rpl = Disc.RplNp1;
    plot3(Rpl*Disc.Xt(1:3:end),Rpl*Disc.Xt(2:3:end), Rpl*Disc.Xt(3:3:end));
end