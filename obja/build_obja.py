from decimate import Decimater
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from visualize import plot_model_with_points_3D





def main():
    """
    Runs the program on the model given as parameter.
    """
    
    np.seterr(invalid = 'raise')
    model = Decimater()
    model.parse_file('example/igea.obj')
    print(len(model.faces))

    with open('example/igea_compressed.obja', 'w') as output:
         model.contract(l=11 , output = output)
    plot_model_with_points_3D(model.mapping , model.faces)
    model.export_to_obj("compressed_model.obj")
    model.uniform_remeshing(m=1)

    print(len(model.faces))

if __name__ == '__main__':
    main()