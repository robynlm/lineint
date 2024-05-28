"""This module provides the LineIntegrate class.

Copyright (C) 2024  Robyn L. Munoz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at: robyn.munoz@yahoo.fr
"""

import numpy as np

class LineIntegrate:
    def __init__(self, xdomain, ydomain, zdomain, idx_start):
        """Initializes parameters for the 3D input grid and 3D grid around zero.
        
        This class provides methods to initialize a 3D input grid and a 3D grid
        centered around zero, and to split a line into segments on these grids
        based on various scenarios, going from the indice of idx_start
        - to another indice in input grid: self.segments_idx_end(idx_end)
        - to a maximum_radius in input grid:self.segments_radius_end(phi, theta)
        and going from zero to when a condition changes sign in grid around 0:
        self.segments_conditional_end(phi, theta, condition, condition_function)
        
        Parameters:
        xdomain (list or tuple of floats): A list or tuple containing the
            minimum value, maximum value, and grid spacing for the x-axis,
            formatted as (xmin, xmax, dx) for the input grid.
        ydomain (list or tuple of floats): A list or tuple containing the
            minimum value, maximum value, and grid spacing for the y-axis,
            formatted as (ymin, ymax, dy) for the input grid.
        zdomain (list or tuple of floats): A list or tuple containing the
            minimum value, maximum value, and grid spacing for the z-axis,
            formatted as (zmin, zmax, dz) for the input grid.
        idx_start (list or tuple of ints): A list or tuple containing the
            starting indices (ixC, iyC, izC) of the line on the input grid.
        
        Attributes:
        xmin, xmax, dx (float): Minimum, maximum and grid spacing on the x-axis.
        ymin, ymax, dy (float): Minimum, maximum and grid spacing on the y-axis.
        zmin, zmax, dz (float): Minimum, maximum and grid spacing on the z-axis.
            Minimum and maximum are valid for only the input grid, grid spacing
            is valid for both input grid and grid around zero.
        ixC, iyC, izC (int): Passed from input.
        maximum_radius (float): If integrating from starting indice
            to given radius, the maximum radius needs to be passed here
            (0 otherwise).
        Lx, Ly, Lz (float): Lengths of the domain along the x, y, and z axes.
            Valid for both input grid and grid around zero.
        Nx, Ny, Nz (int): The number of grid points along the x, y, and z axes.
            Valid for both input grid and grid around zero.
        igridx, igridy, igridz (np.ndarray): Arrays representing the input grid
            points along the x, y, and z axes.
        grid0x, grid0y, grid0z (np.ndarray): Arrays representing the grid points
            centered around zero along the x, y, and z axes.
        gridx_difference, gridy_difference, gridz_difference (int): The
            differences in grid points between the input grid and the grid
            centered around zero.
        
        Notes:
        - The input grid is assumed to be periodic.
        """
        
        # input
        self.xmin, self.xmax, self.dx = xdomain
        self.ymin, self.ymax, self.dy = ydomain
        self.zmin, self.zmax, self.dz = zdomain
        self.ixC, self.iyC, self.izC = idx_start
        
        # input grid
        # upon which the data needs to be periodic
        # otherwise the grid needs to be centered around zero
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        self.Lz = self.zmax - self.zmin
        self.igridx = np.arange(self.xmin, self.xmax, self.dx)
        self.igridy = np.arange(self.ymin, self.ymax, self.dy)
        self.igridz = np.arange(self.zmin, self.zmax, self.dz)
        self.Nx = len(self.igridx)
        self.Ny = len(self.igridy)
        self.Nz = len(self.igridz)
        
        # grid around (0.0, 0.0, 0.0)
        myNx = np.round(self.Lx/(self.dx))
        myNy = np.round(self.Ly/(self.dy))
        myNz = np.round(self.Lz/(self.dz))
        nxmin = (-self.Lx + (myNx % 2) * self.dx) / 2
        nymin = (-self.Ly + (myNy % 2) * self.dy) / 2
        nzmin = (-self.Lz + (myNz % 2) * self.dz) / 2
        self.grid0x = np.arange(nxmin, nxmin + self.Lx, self.dx)
        self.grid0y = np.arange(nymin, nymin + self.Ly, self.dy)
        self.grid0z = np.arange(nzmin, nzmin + self.Lz, self.dz)
        # careful len(grid0) not always = myNx
        self.gridx_difference = (int(myNx/2) - self.ixC)
        self.gridy_difference = (int(myNy/2) - self.iyC)
        self.gridz_difference = (int(myNz/2) - self.izC)
        
        # maximum radius for self.segments_radius_end
        self.maximum_radius = 0
    
    def segments_idx_end(self, idx_end):
        """Splits a line into segments on the input grid up to end index.

        This function adjusts indices of input grid to place the starting
        position of the line at the 0.0 origin. It then wraps around
        `idx_along_r` to compute the segments of a line in a grid around 0,
        starting from the origin and extending to end indice. The grid around 0
        indices are then shifted back to the input grid.

        Parameters:
        idx_end (tuple): A tuple (ix, iy, iz) representing the index of the end
            data point on input grid.

        Returns:
        final_indices (list of tuples): A list of (x, y, z) indices of the data
            points on input grid whose cells are crossed by the line up to the
            conditional endpoint.
        final_segments (np.array): An array containing the lengths of the
            segments between the intersection points up to the end index point.
        phi (float): The azimuthal angle in radians, measured from the x-axis
            in the xy-plane. It is 0 <= phi < 2pi.
        theta (float): The inclination angle in radians, measured from the
            z-axis. It is 0 <= theta < pi.
        """
        if type(idx_end)!=tuple:
            print('ERROR: idx_end needs to be a tuple of three integers',
                  flush=True)
        else:
            if idx_end == (self.ixC, self.iyC, self.izC):
                # if starting point = end point
                return [idx_end], [0.0], 0.0, 0.0
            else:
                
                # === Cartesian coordinates
                # Shift so that starting location is in the center
                # from input grid to my grid around 0
                ixend, iyend, izend = self.idx_igrid_to_grid0(idx_end)
                xend = self.grid0x[ixend]
                yend = self.grid0y[iyend]
                zend = self.grid0z[izend]
                
                # === Spherical coordinates
                rend, phi, theta = self.cart_to_sphe(xend, yend, zend)

                # === Main calculation
                outputs = self.idx_along_r(rend, phi, theta)
                indices, segments, radius, coord = outputs
                
                # === Shift back to starting location
                # from my grid around 0 to input grid
                findices = [self.idx_grid0_to_igrid(idx) for idx in indices]

                # === Cut data to end data point
                i = findices.index(idx_end) + 1
                final_indices = findices[:i]
                final_segments = list(segments[:i])

                # === Fix last segment to end point
                # intersect to intersect => intersect to end point
                numerical_radius = np.sum(final_segments)
                if rend < numerical_radius:
                    final_indices += [findices[i - 1]]
                    final_segments += [rend - radius[i]]
                elif rend > numerical_radius:
                    final_indices += [findices[i]]
                    final_segments += [rend - radius[i]]

                return final_indices, final_segments, phi, theta

    def segments_radius_end(self, phi, theta):
        """Splits a line into segments on a 3D input grid upto a maximum_radius.

        This function wraps around `segments_conditional_end` to compute the
        segments of a line in a 3D grid around 0, starting from the origin,
        extending in the direction specified by the spherical coordinates
        `(phi, theta)`, and ending at the conditional boundary. Here, this
        boundary is defined by `self.maximum_radius`. The indices of the cells
        traversed by this line are then provided according to the input grid.

        Parameters:
        phi (float): The azimuthal angle in radians, measured from the x-axis in
            the xy-plane. It has to be 0 <= phi < 2pi.
        theta (float): The inclination angle in radians, measured from the
            z-axis. It has to be 0 <= theta < pi.

        Returns:
        final_indices (list of tuples): A list of (x, y, z) indices of the data
            points on the input grid whose cells are crossed by the line up to
            the max radius.
        final_segments (np.array): An array containing the lengths of the
            segments between the intersection points up to the max radius.
        """
        x, y, z = np.meshgrid(self.grid0x, self.grid0y, self.grid0z,
                              indexing='ij')
        condition = self.radius_condition(np.array([[x, y, z]]))[0]
        outputs = self.segments_conditional_end(phi, theta, condition,
                                                self.radius_condition)
        indices, final_segments = outputs
        final_indices = [self.idx_grid0_to_igrid(idx) for idx in indices]
        return final_indices, final_segments

    def radius_condition(self, coord):
        """If coordinates are inside (negative) or outside (positive) a sphere.

        Parameters:
        coord (list of tuples or array-like): A list or array of coordinates,
            where each coordinate is a tuple or list of (x, y, z) cartesian
            values on the grid around 0.

        Returns:
        np.ndarray: An array of condition values, where the value is negative
            if the radius is smaller than `self.maximum_radius` and positive
            otherwise.
            
        Note:
        - This funtion's input and output formats are the same as
            scipy.interpolate.RegularGridInterpolator.
        """
        condition = []
        for i in range(len(coord)):
            x, y, z = coord[i]
            condition += [self.radius3D(x, y, z) - self.maximum_radius]
        return np.array(condition)

    def segments_conditional_end(self, phi, theta,
                                 condition, condition_function):
        """Splits a line into segments in a grid around 0 up to a condition.

        This function wraps around `idx_along_r` to compute the segments of a
        line in a 3D grid around 0, starting from the origin and extending in
        the direction specified by the spherical coordinates `(phi, theta)`.
        The end point is adjusted to the boundary where the condition changes
        sign.

        Parameters:
        phi (float): The azimuthal angle in radians, measured from the x-axis in
            the xy-plane. It has to be 0 <= phi < 2pi.
        theta (float): The inclination angle in radians, measured from the
            z-axis. It has to be 0 <= theta < pi.
        condition (array-like): An array of the condition on the grid around 0.
            Radialy going out from the origin (0.0, 0.0, 0.0), the condition
            needs to experience one sign change. This is used as the boundary
            end point.
        condition_function (callable): A function that takes cartesian
            coordinates and returns the condition value (centered around the
            0.0 origin). It's input and output format needs to the same as
            scipy.interpolate.RegularGridInterpolator. This is needed because a
            3D interpolation is required to calculate the boundary point.

        Returns:
        final_indices (list of tuples): A list of (x, y, z) indices of the data
            points whose cells are crossed by the line up to the conditional
            endpoint.
        final_segments (np.array): An array containing the lengths of the
            segments between the intersection points up to the conditional
            endpoint.
        """
        
        # === Maximum radius in phi and theta direction
        Rcell, phicell, thetacell = self.cart_to_sphe(self.dx, self.dy, self.dz)
        thetacell = np.arctan(self.dy/self.dz)
        # if dx = dy = dz then phicell = thetacell = pi / 4
        if theta <= thetacell or np.pi - thetacell <= theta:
            rend = abs(self.Lz / (2*np.cos(theta)))
        elif ((phicell <= phi and phi<= 3*phicell) 
              or (5*phicell <= phi and phi<= 7*phicell)):
            rend = abs(self.Ly / (2*np.sin(phi)*np.sin(theta)))
        else:
            rend = abs(self.Lx / (2*np.cos(phi)*np.sin(theta)))
        rend = np.min([rend, self.radius3D(self.Lx/2, self.Ly/2, self.Lz/2)])

        # === Main calculation
        outputs = self.idx_along_r(rend, phi, theta)
        indices, segments, radius, intersect_coord = outputs
        condition_in_direction = np.array([condition[idx] for idx in indices])

        # === Identify indices closest to boundary
        sign_before_boundary = np.sign(condition_in_direction[0])
        i_before_boundary = np.max(np.where(np.sign(condition_in_direction)
                                            == sign_before_boundary))

        # === Calculate boundary coordinates
        # find intersection point closest to boundary point
        intersect_indices = np.arange(i_before_boundary - 1,
                                      np.min([i_before_boundary + 3,
                                              len(intersect_coord[0])]))
        cb_condition = []
        cb_radius = []
        for i in intersect_indices:
            icoord = [intersect_coord[0][i],
                      intersect_coord[1][i], 
                      intersect_coord[2][i]]
            cb_condition += [condition_function(np.array([icoord]))[0]]
            cb_radius += [self.radius3D(icoord[0], icoord[1], icoord[2])]
            
        # sort to get 2 points closest to boundary
        cb_radius = [i for _,i in sorted(zip(abs(np.array(cb_condition)), 
                                             cb_radius))][:2]
        cb_condition = [i for _,i in sorted(zip(abs(np.array(cb_condition)), 
                                                cb_condition))][:2]

        # calc coord of boundary
        radius_boundary = self.lin_fit_zero([cb_condition[0], cb_condition[1]], 
                                            [cb_radius[0], cb_radius[1]])
        coordboundary = self.sphe_to_cart(radius_boundary, phi, theta)
        xboundary, yboundary, zboundary = coordboundary

        # === Adjust indices and segments to boundary
        last_i = np.max(np.where(radius<radius_boundary))
        final_indices = [indices[i] for i in range(last_i+1)]
        last_segment = np.sqrt((intersect_coord[0][last_i] - xboundary)**2
                               +(intersect_coord[1][last_i] - yboundary)**2
                               +(intersect_coord[2][last_i] - zboundary)**2)
        final_segments = [np.append(segments[:last_i], last_segment)]
        
        return final_indices, final_segments
        
    def idx_along_r(self, rend, phi, theta):
        """Splits a line into segments on a 3D grid around 0.
        
        This is the main calculation of this class.
        
                     Considering
         * | * / *    - data points centered around (0.0, 0.0, 0.0), as provided
        ------/----     by grid0, with their corresponding cell boundaries
         * | / | *    - and a line starting at the origin, whose direction and
        ----/------     size are provided by the input spherical coordinates,
         * / * | *   this function will calculate the coordinates of the
        --/--------  intersections between the line and the cell boundaries.
         0 | * | *   The coordinates of these intersections are passed as
                     outputs of this function.
        
        The line is then split into segments, separated by the intersections.
        Each segment is associated to the corresponding cell being traversed.
        The indices of these cells and their respective segment length are
        passed as outputs of this function.
                
        Parameters:
        rend (float): The radial distance from the origin to the end of the
            line. It has to be 0 <= rend <= max_size_in_box / 2.
        phi (float): The azimuthal angle in radians, measured from the x-axis in
            the xy-plane. It has to be 0 <= phi < 2pi.
        theta (float): The inclination angle in radians, measured from the
            z-axis. It has to be 0 <= theta < pi.

        Returns:
        indices (list of tuples): A list of length Ns of (x, y, z) indices of
            the data points (in the grid around 0) whose cells are traversed
            by the line.
        segments (np.array): An array of length Ns containing the lengths of the
            segments between the intersection points.
        radius (list of floats): A list of length Ns+1 of radial distances of
            the intersection points from the origin.
        coord (list of lists): A list containing three lists (coordx, coordy,
            coordz) of the x, y, and z coordinates of the intersection points.
            This is based on the grid centered around (0.0, 0.0, 0.0).
            There are Ns+1 intersecting points, then there are Ns segments
            between them.
                    
        Notes:
        - The function sorts the intersection points by their radial distances
            from the origin and filters out points beyond the specified radial
            distance `rend`.
        """
        #======================================================================
        # Get the coordinates where the line intersects with the grid around 0
        #======================================================================
        
        # angles
        signx = -1 if phi > np.pi/2 and phi<3*np.pi/2 else 1
        signy =  1 if phi <= np.pi else -1
        signz =  1 if theta <= np.pi /2 else -1
        good_xangle = (theta!=0 and theta!=np.pi
                       and phi!=np.pi/2 and phi!=3*np.pi/2)
        good_yangle = theta!=0 and theta!=np.pi and phi!=0 and phi!=np.pi
        good_zangle = theta!=np.pi/2
        
        # Number of intersections between the line and grid cell boundaries
        rmax = rend + self.radius3D(self.dx, self.dy, self.dz)
        ixmax = int(abs(rmax * np.cos(phi) * np.sin(theta) / self.dx))
        iymax = int(abs(rmax * np.sin(phi) * np.sin(theta) / self.dy))
        izmax = int(abs(rmax * np.cos(theta) / self.dz))

        # == intersect with x lines
        dirx_coordx = np.array([signx * self.dx * (0.5 + i)
                                     for i in range(ixmax) if good_xangle])
        dirx_coordy = dirx_coordx * np.tan(phi)
        dirx_coordz = self.safe_division(self.radius2D(dirx_coordx,
                                                       dirx_coordy),
                                         np.tan(theta))

        # == intersect with y lines
        diry_coordy = np.array([signy * self.dy * (0.5 + i)
                                     for i in range(iymax) if good_yangle])
        diry_coordx = self.safe_division(diry_coordy, np.tan(phi))
        diry_coordz = self.safe_division(self.radius2D(diry_coordx,
                                                       diry_coordy),
                                         np.tan(theta))

        # == intersect with z lines
        dirz_coordz = np.array([signz * self.dz * (0.5 + i)
                                     for i in range(izmax) if good_zangle])
        dirz_coordx = dirz_coordz * np.cos(phi) * np.tan(theta)
        dirz_coordy = dirz_coordz * np.sin(phi) * np.tan(theta)

        # put all the coordinates together
        # start + intersections (end is added in parent function)
        coordx = ([0.0] + list(dirx_coordx)
                  + list(diry_coordx) + list(dirz_coordx))
        coordy = ([0.0] + list(dirx_coordy)
                  + list(diry_coordy) + list(dirz_coordy))
        coordz = ([0.0] + list(dirx_coordz)
                  + list(diry_coordz) + list(dirz_coordz))

        #======================================================================
        # Calculate radius and sort
        #======================================================================
        
        # calc radius of each intersection
        radius = [self.radius3D(xi, yi, zi)
                  for xi, yi, zi in zip(coordx, coordy, coordz)]
        
        # sort the points according to radius size
        coordx = [i for _,i in sorted(zip(radius, coordx))]
        coordy = [i for _,i in sorted(zip(radius, coordy))]
        coordz = [i for _,i in sorted(zip(radius, coordz))]
        radius.sort()

        # cutoff points outside of radius
        i = np.max(np.where(radius < rmax)) + 2
        radius = radius[:i]
        coordx = coordx[:i]
        coordy = coordy[:i]
        coordz = coordz[:i]
        coord  = [coordx, coordy, coordz]

        #======================================================================
        # Calculate size of segments between intersections
        #======================================================================
        
        segments = np.array([np.sqrt((coordx[i+1] - coordx[i])**2
                                     + (coordy[i+1] - coordy[i])**2
                                     + (coordz[i+1] - coordz[i])**2) 
                             for i in range(len(coordx)-1)])

        #======================================================================
        # Get associated index in the grid around 0
        #======================================================================

        # the average between the intersections 
        # will be closer to the position of the data points
        
        # == x indices
        av_coordx = [np.average([coordx[i], coordx[i+1]]) 
                     for i in range(len(coordx) - 1)]
        xidx = [np.argmin(abs(self.grid0x - i)) for i in av_coordx]
        
        # == y indices
        av_coordy = [np.average([coordy[i], coordy[i+1]]) 
                     for i in range(len(coordy) - 1)]
        yidx = [np.argmin(abs(self.grid0y - i)) for i in av_coordy]
        
        # == z indices
        av_coordz = [np.average([coordz[i], coordz[i+1]]) 
                     for i in range(len(coordz) - 1)]
        zidx = [np.argmin(abs(self.grid0z - i)) for i in av_coordz]
        
        # put the indices together
        indices = [(ix, iy, iz) for ix, iy, iz in zip(xidx, yidx, zidx)]
        
        return indices, segments, radius, coord


################################################################################
################################################################################
################################################################################
#########
#########          HELPER FUNCTIONS
#########
################################################################################
################################################################################
################################################################################

        
    def safe_division(self, numerator, denominator):
        """Perform safe division.
        
        This checks if the numerator or denominator are zero
        and returns their ratio when they are non-zero, and zero otherwise.
        Even though numerator / zero =/= zero, I do this to not introduce
        NaN values and this is valid where I call this function,
        e.g. theta = arccos( z / r ), if r = 0 then theta = 0.
        
        Parameters:
        numerator (float or array-like): The numerator of the division.
        denominator (float or array-like): The denominator of the division.

        Returns:
        float or array-like: The result of the division.
        """
        num0 = numerator == 0
        den0 = denominator == 0
        if isinstance(numerator, float):
            ratio = 0.0 if den0 else numerator/denominator
        else: # it's an array
            ratio = np.divide(numerator, denominator,
                              out = np.zeros_like(numerator),
                              where = ~den0)
            ratio[np.where(np.logical_and(num0, den0))] = 0.0
        return ratio
        
    def lin_fit_zero(self, f, r):
        """Perform a linear fit to find the zero-crossing point of a line.

        Parameters:
        f (list or array-like): A list or array containing the `f` values
            (dependent variable) of the two points.
        r (list or array-like): A list or array containing the `r` values
            (independent variable) of the two points.

        Returns:
        float or np.ndarray: The `r` value where the linear fit crosses zero.
            If the input `r` values are identical, it returns the shared `r`
            value as an `np.ndarray`.

        Notes:
        - This function assumes that `f` and `r` are lists or arrays with
            exactly two elements each.
        """
        if r[0]==r[1]:
            return np.array(r[0])
        else:
            a = self.safe_division(f[0]-f[1], r[0]-r[1])
            b = f[0]-r[0]*a
            return self.safe_division(-b, a)
            
    def cart_to_sphe(self, x, y, z):
        """Convert Cartesian coordinates to spherical coordinates.

        Parameters:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate of the point.

        Returns:
        tuple: A tuple containing the spherical coordinates (r, phi, theta).
            r (float): The radial distance from the zero origin.
            phi (float): The azimuthal angle in radians, measured from the
                x-axis in the xy-plane. It is 0 <= phi < 2pi.
            theta (float): The inclination angle in radians, measured from the
                z-axis. It is 0 <= theta < pi.
        """
        # radius
        r = self.radius3D(x, y, z)
        # azimuth
        phi = np.arccos(self.safe_division(x, self.radius2D(x, y)))
        if y<0:
            phi = 2*np.pi - phi
        if x==0 and y==0:
            phi = 0
        # inclination
        theta = np.arccos(self.safe_division(z, r))
        if r==0:
            theta = 0
        return r, phi, theta

    def sphe_to_cart(self, r, phi, theta):
        """Convert spherical coordinates to Cartesian coordinates."""
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)
        return x, y, z
            
    def idx_igrid_to_grid0(self, idx):
        """Periodically convert input grid indices to grid around 0 indices."""
        ixaround0 = (idx[0] + self.gridx_difference) % self.Nx
        iyaround0 = (idx[1] + self.gridy_difference) % self.Ny
        izaround0 = (idx[2] + self.gridz_difference) % self.Nz
        return (ixaround0, iyaround0, izaround0)
        
    def idx_grid0_to_igrid(self, idx):
        """Periodically convert grid around 0 indices to input grid indices."""
        ix = (idx[0] - self.gridx_difference) % self.Nx
        iy = (idx[1] - self.gridy_difference) % self.Ny
        iz = (idx[2] - self.gridz_difference) % self.Nz
        return (ix, iy, iz)

    def shift_igrid_to_grid0(self, data):
        """Shifts a 3D np.array around the zero in a periodic grid."""
        databig = np.append(data[-self.gridx_difference:,:,:],
                            data[:-self.gridx_difference,:,:], axis=0)
        databig = np.append(databig[:,-self.gridy_difference:,:],
                            databig[:,:-self.gridy_difference,:], axis=1)
        databig = np.append(databig[:,:,-self.gridz_difference:],
                            databig[:,:,:-self.gridz_difference], axis=2)
        return databig
        
    def radius2D(self, x, y):
        """Compute 2D radius from zero."""
        return np.sqrt((x)**2 + (y)**2)
        
    def radius3D(self, x, y, z):
        """Compute 3D radius from zero."""
        return np.sqrt((x)**2 + (y)**2 + (z)**2)
        
    def radius3D_igrid_idx(self, idx):
        """Compute 3D radius at the input grid index from center/starting point.

        Parameters:
        idx (tuple): A tuple (ix, iy, iz) representing the input grid index.

        Returns:
        float: Euclidian distance from the center/starting point in input grid.
        """
        if idx == (self.ixC, self.iyC, self.izC):
            return 0.0
        else:
            # instead of measuring on input grid I use grid around 0,
            # to go through the periodic boundaries
            ix0, iy0, iz0 = self.idx_igrid_to_grid0(idx)
            x = self.grid0x[ix0]
            y = self.grid0y[iy0]
            z = self.grid0z[iz0]
            return self.radius3D(x, y, z)
