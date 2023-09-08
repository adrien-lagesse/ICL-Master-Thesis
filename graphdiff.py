import torch
import networkx as nx
import matplotlib.pyplot as plt
import math

class Graph:
    V: int = None
    device: str = None

    adjacency_matrix: torch.FloatTensor = None

    D: torch.FloatTensor = None
    augmented_D: torch.FloatTensor = None

    laplacian: torch.FloatTensor = None
    normalized_laplacian: torch.FloatTensor = None
    augmented_normalized_laplacian: torch.FloatTensor = None

    def __init__(self, 
                 adj_matrix: torch.FloatTensor, 
                 device = 'cpu') -> None:
        """
        Create a Graph object that implements the different notions introduced in the report

        ## Args:
        
        - adj_matrix: Adjacency Matrix representing the structure of the graph
        - device: 'cpu' or 'cuda'. Use cuda for GPU optimization
        """
        
        self.V = adj_matrix.shape[0]
        self.device = device

        self.adjacency_matrix = adj_matrix
        self.adjacency_matrix = self.adjacency_matrix.to(device)

        self.D = torch.diag(self.adjacency_matrix.sum(dim=0))
        self.augmented_D = torch.diag(1+self.adjacency_matrix.sum(dim=0))

        self.D_sinv = torch.diag(1/torch.sqrt(self.adjacency_matrix.sum(dim=0)))
        self.augmented_D_sinv = torch.diag(1/torch.sqrt(1+self.adjacency_matrix.sum(dim=0)))

        self.laplacian = self.D - self.adjacency_matrix
        self.normalized_laplacian = self.D_sinv@self.laplacian@self.D_sinv
        self.augmented_normalized_laplacian = self.augmented_D_sinv@self.laplacian@self.augmented_D_sinv
    
    def get_frequencies(self, 
                        type : str) -> torch.FloatTensor:
        """
        Compute the graph frequencies and order them in ascending order.
        ## Args:
        - type: 'normal', 'normalized' or 'augmented_normalized'. State which Laplacian to use

        ## Returns:
        FloatTensor of shape (V,) with the frequencies in ascending order.
        """

        lap = None
        if type == 'normal':
            lap = self.laplacian
        elif type == 'normalized':
            lap = self.normalized_laplacian
        elif type == 'augmented_normalized':
            lap = self.augmented_normalized_laplacian
        else:
            raise RuntimeError("Wrong Laplacian type")
                
        (freqs, _) = torch.linalg.eigh(lap)

        return freqs
    
    def get_eigensignals(self, 
                        type : str) -> torch.FloatTensor:
        """
        Compute the graph eigenvalues of the graph.
        ## Args:
        - type: 'normal', 'normalized' or 'augmented_normalized'. State which Laplacian to use

        ## Returns:
        FloatTensor of shape (V,V)
        """

        lap = None
        if type == 'normal':
            lap = self.laplacian
        elif type == 'normalized':
            lap = self.normalized_laplacian
        elif type == 'augmented_normalized':
            lap = self.augmented_normalized_laplacian
        else:
            raise RuntimeError("Wrong Laplacian type")
        
        (_, eigensignals) = torch.linalg.eigh(lap)

        return eigensignals.T
    
    def ft(self, signal:torch.FloatTensor, type:str):
        """
        Compute the graph Fourier transform of 'signal'.

        ## Args:
        - signal: Signal of shape (V,l).
        - type: 'normal', 'normalized' or 'augmented_normalized'. State which Laplacian to use

        ## Returns:
        FloatTensor of shape (V,l)
        """
        s = signal.to(self.device)
        P = self.get_eigensignals(type)
        return P@s

    def visualize(self):
        """
        Graph visualisation. 
        """
        G = nx.from_numpy_array(self.adjacency_matrix.cpu().numpy())
        nx.draw(G)
        plt.show()
    
    def visualize_signal(self, 
                         signal: torch.FloatTensor, 
                         color_normalization = None):
        """
        Visualize 'signal' on the graph G.

        ## Args:
        - signal: FloatTensor of shape (V,1)
        - color_normalization: function taking a value and returning a RGB color
        """
        assert signal.shape == (self.V,1), "Signal must be of shape (V,1)"

        node_weights = signal.flatten().cpu()

        G = nx.from_numpy_array(self.adjacency_matrix.cpu().numpy())

        if color_normalization is None:
            color_normalization = lambda x: (x-torch.min(node_weights))/(torch.max(node_weights)-torch.min(node_weights))

        nx.draw(G, node_color = [(0,color_normalization(x),color_normalization(x)) for x in node_weights.numpy()])
        plt.show()
    
    def visualize_ft(self, 
                     signal: torch.FloatTensor,  
                     type= str):
        """
        Create a Bar Chart representation of the signal by frequency.
        ## Args:
        - signal: FloatTensor of shape (V,1)
        - type: 'normal', 'normalized' or 'augmented_normalized'. State which Laplacian to use
        """
        assert signal.shape == (self.V,1), "Signal must be of shape (V,1)"

        s = signal.flatten().to(self.device)
        P = self.get_eigensignals(type)
        coeffs = P@s
        plt.bar([i for i in range(len(coeffs))], coeffs.cpu().numpy())
        plt.title(f"Fourier Transform Decomposition ({type})")
        plt.xlabel("Frequencies index")
        plt.ylabel("Coefficient")
        plt.show()
            
    def visualize_energy_decomposition(self, signal: torch.FloatTensor,  type= str):
        """
        Create a Bar Chart representation of the signal Dirichlet energy by frequency.
        ## Args:
        - signal: FloatTensor of shape (V,1)
        - type: 'normal', 'normalized' or 'augmented_normalized'. State which Laplacian to use
        """
        assert signal.shape == (self.V,1), "Signal must be of shape (V,1)"

        s = signal.flatten().to(self.device)
        P = self.get_eigensignals(type)
        coeffs = P@s
        plt.bar([i for i in range(len(coeffs))], coeffs.cpu().numpy()**2)
        plt.title(f"Dirichlet Energy Decomposition ({type})")
        plt.xlabel("Frequencies index")
        plt.ylabel("Energy")
        plt.show()
    
    def energy(self, 
               signal: torch.FloatTensor,  
               type= str) -> float:
        """
        Compute the signal Dirichlet energy.

        ## Args:
        - signal: FloatTensor of shape (V,l)
        - type: 'normal', 'normalized' or 'augmented_normalized'. State which Laplacian to use.

        ## Returns:
        Signal Dirichlet Energy
        """
        lap = None
        if type == 'normal':
            lap = self.laplacian
        elif type == 'normalized':
            lap = self.normalized_laplacian
        elif type == 'augmented_normalized':
            lap = self.augmented_normalized_laplacian
        else:
            raise RuntimeError("Wrong Laplacian type")
        
        s = signal.to(self.device)
        return math.sqrt(float(torch.trace(s.T@lap@s)))
