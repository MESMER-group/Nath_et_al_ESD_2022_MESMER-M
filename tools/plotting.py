#!/usr/bin/env python
# -*- coding: utf-8 -*-


#################
## Colorbar for imshow / pcolormesh from Loris Foresti (adjusted)
#################            
            
from matplotlib.colors import from_levels_and_colors 
from pylab import get_cmap

def smart_colormap(clevs, name='jet', extend='both',max_col='None',min_col='None'):
    """ Grab the colors to extend the colorbar from the colormap.
	
	Keyword argument:
		- clevs: vector / list with ticks for the colorbar

	Optional arguments:
		- name: cmap name
		- extend: whether or not to extend the colorbar on top&/bottom
		- max_col: replaces maximum color with color of different cmap (3 vals -> idx 1, hardcoded)
		- min_col: replaces minimum color with color of different cmap (3 vals -> idx 1, hardcoded)

	output:
		- cmap
		- norm

	Remarks:
		- ATTENTION max_col / min_col atm hardcoded as idx 1 of cmap with 3 entries -> could adapt later

    """
    
    # Define number of colors
    if extend == 'both':
        nrColors = len(clevs)+1
    elif (extend == 'min') | (extend == 'max'):
        nrColors = len(clevs)
    elif (extend == 'neither'):
        nrColors = len(clevs)-1
    else:
        nrColors = len(clevs)-1
        extend = 'neither'

    if (max_col != 'None'):
        nrColors -=1
        
    if (min_col != 'None'):
        nrColors -=1  

    # Get colormap
    cmap = get_cmap(name, nrColors)
    
    
    # Get the list of colors
    colors = []
    for i in range(0, nrColors):
        colors.append(cmap(i/(nrColors-1)))

    if (min_col != 'None'):
        cmap = get_cmap(min_col,3)
        colors = [cmap(1)]+colors
    
    if (max_col != 'None'):
        cmap = get_cmap(max_col,3)
        colors = colors+[cmap(1)]
    
    # Use utility function to get cmap and norm at the same time
    cmap, norm = from_levels_and_colors(clevs, colors, extend=extend)

    return(cmap, norm)

#!/usr/bin/env python
# Copyright: This document has been placed in the public domain.

"""
Taylor diagram (Taylor, 2001) implementation.

Note: If you have found these software useful for your research, I would
appreciate an acknowledgment.
"""

__version__ = "Time-stamp: <2018-12-06 11:43:41 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

import numpy as NP
import matplotlib.pyplot as PLT


class TaylorDiagram(object):
    """
    Taylor diagram.

    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.6), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.

        Parameters:

        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = NP.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = NP.pi
            rlocs = NP.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = NP.pi/2
        tlocs = NP.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = PLT.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = NP.linspace(0, self.tmax)
        r = NP.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'r')
        
        for i_r in [0.2,0.4,0.6,0.8,1,1.2,1.4]:
            if i_r==self.refstd:
                continue
            t = NP.linspace(0, self.tmax)
            r = NP.zeros_like(t) + i_r
            self.ax.plot(t, r, 'k--')
        
        for i_t in tlocs:
            r = NP.linspace(self.smin, self.smax)
            t = NP.zeros_like(r) + i_t
            self.ax.plot(t, r, 'b--')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(NP.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = NP.meshgrid(NP.linspace(self.smin, self.smax),
                             NP.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = NP.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*NP.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours
