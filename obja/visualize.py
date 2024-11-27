import pyvista as pv
import numpy as np




def barycentric_to_cartesian(triangle, bary_coords):
    """
    Convertit les coordonnées barycentriques en coordonnées cartésiennes.
    """
    A, B, C = triangle
    u, v, w = bary_coords
    return u * A + v * B + w * C

def plot_model_with_points_3D(mapping, triangles):
    """
    Affiche le modèle 3D défini par des triangles et ajoute des points exprimés en coordonnées barycentriques.
    
    :param mapping: Dictionnaire des sommets où chaque sommet peut être un tuple de coordonnées cartésiennes
                    ou des coordonnées barycentriques avec des indices de sommets.
    :param triangles: Liste d'objets triangles, chaque triangle ayant les attributs `a`, `b`, `c` pour ses sommets.
    """
    plotter = pv.Plotter()
    
    # Ajouter les triangles au plotter
    for triangle in triangles:
        # Extraire les coordonnées cartésiennes des sommets du triangle
        coords = [mapping[triangle.a], mapping[triangle.b], mapping[triangle.c]]
        
        # Créer une surface triangulaire
        surface = pv.PolyData(np.array(coords), faces=[3, 0, 1, 2])
        plotter.add_mesh(surface, color='lightblue', show_edges=True, edge_color='black', line_width=1)
    
    # Ajouter les points barycentriques
    for v_index, data in mapping.items():
        if isinstance(data, tuple):  # Vérifier si le sommet est défini par des coordonnées barycentriques
            # Récupérer le triangle associé
            triangle_indices = [data[3], data[4], data[5]]
            triangle = [mapping[triangle_indices[0]], mapping[triangle_indices[1]], mapping[triangle_indices[2]]]
            
            # Convertir les coordonnées barycentriques en cartésiennes
            cartesian_point = barycentric_to_cartesian(triangle, data[:3])
            
            # Ajouter le point au plotter
            plotter.add_mesh(pv.PolyData(cartesian_point), color='red', point_size=7)
    
    
    # Afficher la visualisation
    plotter.show()
