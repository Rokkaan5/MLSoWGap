from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np

def density_scatter( x , y, line_z, ax = None, sort = True, bins = 20, nlevels=10, plot_cbar=True, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = False )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0
    
    z[np.where(z<=0.0)] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    #x & y values for plotting line of best fit for the model
    a, b = np.polyfit(x, y, 1)
    fit_line_x = [-1*line_z,line_z]
    fit_line_y = a*np.array(fit_line_x)+b
    
    cs = ax.scatter( x, y, c=z, marker='.', cmap='plasma', alpha=0.8, **kwargs )     # 'marker' was added by JK
    ax.plot([-1*line_z,line_z], [-1*line_z,line_z], 'forestgreen', linewidth=3., label='R = 1.0')
    ax.plot(fit_line_x, fit_line_y, linestyle='--',color='red', 
            linewidth=2., label='R = {0:.2f} (model)'.format(pearsonr(x,y)[0]))
    #norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cticks = list(np.linspace(0,np.ceil(len(x)/10.)*10,nlevels))
    
    # print(z.max())
    cbar = plt.colorbar(cs,ax=ax,ticks=cticks)
    cbar.ax.set_yticklabels(["{:.1f}".format(i) for i in cbar.get_ticks()])
    cbar.set_label('# per bin', fontsize=12, labelpad=0.75)
    if not plot_cbar:
        cbar.remove()
    return ax
