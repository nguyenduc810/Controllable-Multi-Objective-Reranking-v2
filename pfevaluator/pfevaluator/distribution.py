from pfevaluator.pfevaluator.root import Root
from numpy import sum
import numpy as np


class Metric(Root):
    def __init__(self, pareto_front=None, reference_front=None, **kwargs):
        super().__init__(pareto_front, reference_front)
        ## Other parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def spacing(self, fronts = None):
        dominated_list = self.find_reference_front(fronts)
        n = dominated_list.shape[0]
        # print(dominated_list)
        # print(n)
        d_S = []
        for i in range(n):
            d_min = 10e10
            for j in range(n):
                if i != j:
                    # print(dominated_list[i])
                    d  = np.linalg.norm(dominated_list[i] - dominated_list[j], ord=1)
                    if d < d_min:
                        d_min = d
            d_S.append(d_min)
        d_S = np.array(d_S)
        d_mean = np.mean(d_S)
        SP = np.sqrt(np.linalg.norm(d_mean - d_S, ord=2)/(n-1))
        return SP

    def Hole_relative_size(self, fronts = None):
        dominated_list = self.find_reference_front(fronts)
        n = dominated_list.shape[0]
        d_S = []
        for i in range(n):
            d_min = 10e10
            for j in range(n):
                if i != j:
                    # print(dominated_list[i])
                    d  = np.linalg.norm(dominated_list[i] - dominated_list[j], ord=1)
                    if d < d_min:
                        d_min = d
            d_S.append(d_min)
        d_S = np.array(d_S)
        d_mean = np.mean(d_S)
        d_max = np.max(d_S)
        return d_max/d_mean
    
    def pareto_ratio(self, fronts=None):
        dominated_list = self.find_reference_front(fronts)
        return len(dominated_list)/len(fronts)

    def calculate_hypervolume(self,fronts=None):
        """
        Calculate the hypervolume of a set of points with respect to the reference point (0, 0).

        Parameters:
        - points: numpy array, shape (n, 2), where n is the number of points.

        Returns:
        - hypervolume: float, the calculated hypervolume.
        """
        # Sort the points based on the first coordinate (ascending order)
        points = self.find_reference_front(fronts)
        sorted_points = points[np.argsort(points[:, 0])]

        # Initialize hypervolume to 0
        hypervolume = 0.0

        # Iterate through sorted points and calculate the contribution to the hypervolume
        for i in range(len(sorted_points)):
            width = sorted_points[i, 0]
            height = sorted_points[i, 1]
            hypervolume += width * height

        return hypervolume

