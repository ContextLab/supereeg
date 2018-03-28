bo.get_locs() #should always return whatever locations we'll want to plot or use in reconstructions
bo.get_data() #should always return whatever data we'll want to use or plot

Brain(anything) #--> that should always return a brain object
Brain(x) #(x is a brain object) --> should return x...not a copy of x

if Brain(x)[3] > 4:
    do something
else:
    do something else


ni.plot_glass_brain(Nifti(x))
