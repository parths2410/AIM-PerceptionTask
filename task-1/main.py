from collections import deque
import math
from matplotlib import pyplot as plt
import numpy as np

from utils import *

class Line:
    """
    Represents a 2D line defined by a point and an angle. Only used for visualization.
    """
    def __init__(self, point, angle):
        self.point = point
        self.angle = angle
        self.a = math.tan(np.deg2rad(angle))
        self.b = -1
        self.c = point[1] - self.a*point[0]

    def signed_distance(self, point):
        """Computes the signed distance from a point to the line."""
        distance = self.a*point[0] + self.b*point[1] + self.c / math.sqrt(self.a**2 + self.b**2)
        return distance
        
    def __str__(self):
        return f"Line: {self.a:.2f}x + {self.b:.2f}y + {self.c:.2f} = 0 :: Angle: {self.angle:.2f} :: Point: {self.point}"
        
    def y(self, x):
        """Computes the y-coordinate given an x-coordinate on the line."""
        return - (self.a*x + self.c) / self.b
    
    def view(self, ax):
        """Plots the line on the given axis."""
        x = np.linspace(0, 5, 100)
        ax.plot(x, self.y(x))
    
class Cop:
    """
    Represents a Cop in the city with a location, orientation and field of view.
    """
    def __init__(self, id, block, orientation, fov):
        self.id = id
        self.block = block
        self.location = (block[1] + 0.5, -block[0] - 0.5)
        self.orientation = orientation
        self.fov = fov
        self.view_angles = [roll_angle(orientation - fov/2), roll_angle(orientation + fov/2)]
        
        self.view_lines = [Line(self.location, angle) for angle in self.view_angles]

    def check_in_view(self, block):
        """Checks if a given block is within the field of view of the cop."""
        center_x = block[1] + 0.5
        center_y = -block[0] - 0.5
        
        corners = [
            (center_x - 0.5, center_y - 0.5),
            (center_x + 0.5, center_y - 0.5),
            (center_x + 0.5, center_y + 0.5),
            (center_x - 0.5, center_y + 0.5)
        ]                
        
        angles = [self.angle_to(corner) for corner in corners]
        corner_in_view = [is_angle_in_fov(angle, self.view_angles[0], self.view_angles[1]) for angle in angles]
               
        if any(corner_in_view) or (block == self.block):
            return True
        
        # TODO : edgecase :: when fov is too small -- no corners are in the fov, but the block is in the fov        
        for i in range(4):
            A, B = corners[i], corners[(i+1) % 4]
                        
            A_ = (A[0] - self.location[0], A[1] - self.location[1])
            B_ = (B[0] - self.location[0], B[1] - self.location[1])
            
            alpha = self.view_angles[0]
            u = (np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha)))
            if lineseg_ray_intersection(A_, B_, u):
                return True
            
            alpha = self.view_angles[1]
            v = (np.cos(np.deg2rad(alpha)), np.sin(np.deg2rad(alpha)))
            if lineseg_ray_intersection(A_, B_, v):
                return True
            
        return False
    
    def angle_to(self, point):
        """Computes the angle from the cop to a given point."""
        angle_rad = math.atan2(point[1] - self.location[1], point[0] - self.location[0])
        return roll_angle(np.rad2deg(angle_rad))

    
    def view(self, ax):
        """Plots the cop's location and field of view."""
        ax.plot(self.location[0], self.location[1], 'go')
        for line in self.view_lines:
            line.view(ax)
    
    def __str__(self):
        return f"Cop ::: Block: {self.block} :: Orientation: {self.orientation} :: FOV: {self.fov}"

class Thief:
    """
    Represents the Thief in the city with a location.
    """
    def __init__(self, block):
        self.block = block
        self.location = (block[1] + 0.5, -block[0] - 0.5)
        
    def view(self, ax):
        corners = [
            (self.location[0] - 0.5, self.location[1] - 0.5), 
            (self.location[0] + 0.5, self.location[1] - 0.5),
            (self.location[0] + 0.5, self.location[1] + 0.5),
            (self.location[0] - 0.5, self.location[1] + 0.5)
        ]
        
        # plot the box created by corners
        corners.append(corners[0])
        corners = np.array(corners)
        ax.plot(corners[:, 0], corners[:, 1], 'r')
        
class City:
    """
    Represents the game environment with a grid, cops, and a thief.
    """
    def __init__(self, grid, cop_orientations, cop_fovs):
        self.grid = grid
        self.size = (len(grid), len(grid[0]))
        self.cops = []
        self.thief = None
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if grid[i][j] == 'T':
                    self.thief = Thief((i, j))
                elif grid[i][j] > 0:
                    cop_idx = grid[i][j] - 1
                    cop = Cop(cop_idx + 1, (i, j), cop_orientations[cop_idx], cop_fovs[cop_idx])
                    self.cops.append(cop)
                
        assert self.thief is not None, "Thief not found in the grid"

    def show_city(self):
        """
        Print out the city grid with the cops and thief.
        """
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if (i, j) == self.thief.block:
                    print('T', end=' ')
                elif any([cop.block == (i, j) for cop in self.cops]):
                    print('C', end=' ')
                else:
                    print('.', end=' ')
            print()
    
    def view(self):
        """
        Plot the city environment.
        """
        x_len = self.size[1]
        y_len = -self.size[0]
        fig, ax = plt.subplots()
        ax.set_xlim(0, x_len)
        ax.set_ylim(y_len, 0)    
        ax.grid(True)
        for cop in self.cops:
            cop.view(ax)
        self.thief.view(ax)
        plt.show()
        
    def visibility(self, block):
        """Returns  a list of cop ids that can see the given block"""
        return [cop.id for cop in self.cops if cop.check_in_view(block)]
    
    def generate_visibility_gird(self):
        """
        Generates a grid showing which cops can see each blocks.
        """
        self.visibility_grid = [[self.visibility((i, j)) for j in range(self.size[1])] for i in range(self.size[0])]
    
    def valid_block(self, block):
        """Check if a block is within the city grid."""
        return block[0] >= 0 and block[0] < self.size[0] and block[1] >= 0 and block[1] < self.size[1]
    
    def find_closest_invisible_block(self, block):
        """
        Finds the closest invisible block to the given block using breadth-first search.
        """
        q = deque()

        visited = [[False for _ in range(self.size[1])] for _ in range(self.size[0])]
        visited[block[0]][block[1]] = True

        q.append(block)

        steps = [[-1, 0], [1, 0], [0, -1], [0, 1]]

        while q:
            current = q.popleft()

            if self.visibility_grid[current[0]][current[1]] == []:
                return current

            for step in steps:
                new_i = current[0] + step[0]
                new_j = current[1] + step[1]

                if self.valid_block((new_i, new_j)) and not visited[new_i][new_j]:
                    visited[new_i][new_j] = True
                    q.append((new_i, new_j))

        return None

def thief_and_cops(grid, orientations, fovs):
    # Create the game environment
    game = City(grid, orientations, fovs)

    # Find which cops can see the thief
    cops_viewing_thief = game.visibility(game.thief.block)

    # Generate the visibility grid
    game.generate_visibility_gird()
    # Find the closest invisible block to the thief 
    safe_block = game.find_closest_invisible_block(game.thief.block)

    return cops_viewing_thief, list(safe_block)

if __name__=="__main__":
    grid = [[0, 0, 0, 0, 0], \
            ['T', 0, 0, 0, 2], \
            [0, 0, 0, 0, 0], \
            [0, 0, 1, 0, 0],\
            [0, 0, 0, 0, 0]]
    
    orientations = [180, 150]
    fovs = [60, 60]
    
    cops_viewing_thief, safe_block = thief_and_cops(grid, orientations, fovs)
    
    print(cops_viewing_thief, safe_block)