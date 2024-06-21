
import torch
import numpy as np

class DataSampler:
    def __init__(self, paths, num_pts, num_coords):
        
        if isinstance(paths, str):
            self.paths = [paths]
        else:
            self.paths = paths
        
        if isinstance(num_pts, int):
            self.num_pts = [num_pts]
        else:
            self.num_pts = num_pts
            
        assert len(self.num_pts) == len(self.paths)
        
        self.num_coords = num_coords
        
    def __call__(self, full = False):
        
        pts = []
        data = []
        
        for path, N in zip(self.paths, self.num_pts):
            dataset = np.load(path, mmap_mode='r')
            
            if full:
                idx = np.arange(dataset.shape[0])
            else:
                idx = np.random.choice(dataset.shape[0], N)
                
            loaded = torch.tensor(dataset[idx], dtype=torch.float32)
            
            loaded_pts = loaded[:, :self.num_coords]
            loaded_data = loaded[:, self.num_coords:]
            
            if len(loaded_pts.shape) == 1:
                loaded_pts = loaded_pts.reshape(-1, 1)
            if len(loaded_data.shape) == 1:
                loaded_data = loaded_data.reshape(-1, 1)
            
            pts.append(loaded_pts)
            data.append(loaded_data)
        
        if len(pts) == 1 and len(data) == 1:
            return pts[0], data[0]
        return pts, data