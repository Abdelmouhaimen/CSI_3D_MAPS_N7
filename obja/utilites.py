
import numpy as np
from sklearn.decomposition import PCA
from shapely.geometry import Polygon, MultiPoint, LineString
from shapely.ops import triangulate
from  visualize import  barycentric_to_cartesian
from scipy.spatial import Delaunay






def assign_barycentric_coords(triangle, point):
    """
    Calcule les coordonnées barycentriques d'un point par rapport à un triangle donné.
    
    :param triangle: Un tuple contenant trois points représentant les sommets du triangle.
    :param point: Le point pour lequel on souhaite calculer les coordonnées barycentriques.
    :return: Les coordonnées barycentriques (u, v, w) et le dénominateur utilisé dans les calculs.
    """
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, point - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    
    u = 1 - v - w
    return u, v, w,denom


def find_point_in_triangle( triangles, point):
    """
    Trouve si un point donné est situé dans un des triangles fournis.
    
    :param triangles: Liste de triangles, où chaque triangle est défini par trois points.
    :param point: Le point à tester.
    :return: L'indice du triangle contenant le point, ou la fin de la liste si le point n'est pas trouvé.
    """
    for i, triangle in enumerate(triangles):
        
        u, v, _, _ = assign_barycentric_coords(triangle, point)
        tolerance = 1e-10
        if u >= -tolerance and v >= -tolerance and (u + v) <= 1 + tolerance:
            return i
    return -1  # Retourne -1 si le point n'est trouvé dans aucun triangle


def lstsq_quadrics_fitting(pos_xyz):
    """
    Adapter un ensemble donné de points 3D (x, y, z) à une quadrique de l’équation  ax^2 + by^2 + cxy + dx + ey + f = z 
    paramètre pos_xyz : un tableau numpy à deux dimensions contenant les coordonnées des points 
    return: Coefficients de la quadrique (a, b, c, d, e, f).
    """

    A = np.ones((pos_xyz.shape[0], 6))
    A[:, 0:2] = np.square(pos_xyz[:, 0:2], pos_xyz[:, 0:2])
    A[:, 2] = np.multiply(pos_xyz[:, 0], pos_xyz[:, 1])
    A[:, 3:5] = pos_xyz[:, 0:2]
    
    Z = pos_xyz[:, 2]
    
    X, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    
    return X



def make_pca(data, origin, n_components):
    """
    Calculer l’ACP (Analyse en Composantes Principales) sur les données fournies
    :param n_components : nombre de composantes principales pour l’ACP
    :param data : tableau contenant les données
    :return: projection des données sur  les nouvelles composantes principales.
    """
    x = np.vstack((np.array(data), origin))
    xc = x - origin
    #xc = xc.to_numpy()
    pca = PCA(n_components=n_components)
    pca.fit(xc)
    w = pca.components_
    new_coord = xc @ np.transpose(w)
    return new_coord

def separate_upper_lower_parts(cycle:list , path:list) :
    """
    Sépare un cycle en deux parties (supérieure et inférieure) le long d'un chemin donné.
    :param cycle: Liste représentant le cycle.
    :param path: Liste représentant le chemin entre deux sommets v1 et v2.
    :return: Les listes des parties supérieure et inférieure.
    """
    v1 = path[0] 
    v2 = path[1]
    lower = []
    upper = []

    next = cycle.index(v1) 
    while next != cycle.index(v2) :
        upper.append(next)
        next = next+1 if next< len(cycle) -1 else 0
    upper.append(next)

    next = cycle.index(v2) 
    while next != cycle.index(v1) :
        lower.append(next)
        next = next+1 if next< len(cycle) -1 else 0
    lower.append(next)
    return upper , lower



def straighten_along_xaxis(cycle, path, positions_2D):
    coord1 = positions_2D[cycle.index(path[0]), :]
    coord2 = positions_2D[cycle.index(path[1]), :]
    
    # Calcul de l'angle entre le vecteur défini par les deux points et l'axe des X
    angle = -np.arctan2(coord2[1] - coord1[1], coord2[0] - coord1[0])
    
    # Matrice de rotation pour aligner le vecteur (coord1 -> coord2) avec l'axe X
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Appliquer la rotation sur toutes les positions 2D
    positions_2D_rotated = (rotation_matrix @ positions_2D.T).T
    
    return positions_2D_rotated

def retrangulate_upper_lower_parts(path, cycle ,  positions_2D) :
    """
    Retriangule les parties supérieure et inférieure séparées le long d'un chemin donné.
    :param path: Le chemin qui sépare les parties supérieure et inférieure.
    :param cycle: Le cycle contenant les sommets.
    :param positions_2D: Positions 2D des sommets.
    :return: Liste de triangles pour les nouvelles parties retriangulées.
    """
    upper , lower =  separate_upper_lower_parts(cycle, path)
    lower_part =  positions_2D[lower ,:]
    upper_part = positions_2D[upper ,:]
    boundary_l = Polygon(np.vstack([lower_part,lower_part[0,:]]))
    boundary_u = Polygon(np.vstack((upper_part,upper_part[0,:])))
    triangles  =  [tri for tri in triangulate(boundary_u) if tri.within(boundary_u)] + [tri for tri in triangulate(boundary_l) if tri.within(boundary_l)]
    return triangles


"""
# Objectif : Calculer les k plus proches voisins pour chaque sommet du maillage.
# Paramètres :
# - profondeur_k (int) : Profondeur pour la recherche des k plus proches voisins.
# - nsommet (int) : Nombre de sommets dans le maillage.
# voisins_directs (dictionnaire) : Dictionnaire contenant les voisins directs de chaque sommet.
# Retour :
# - knn_dict (dictionnaire) : Dictionnaire où chaque clé est l'indice d'un sommet et la valeur 
# correspondante est un ensemble contenant les indices de ses k plus proches voisins."""
def compute_knn(profondeur_k , voisins_directs,sommet):
    return get_k_neighbors(voisins_directs, profondeur_k, sommet, set())
"""
# Objectif : Fonction auxiliaire pour obtenir les k plus proches voisins d'un sommet donné.
# Paramètres :
# - voisins_directs (dictionnaire) : Dictionnaire contenant les voisins directs de chaque sommet.
# - profondeur_k (int) : Profondeur pour la recherche des k plus proches voisins.
# - point_index (int) : Indice du sommet pour lequel on cherche les k plus proches voisins.
# - n_set (ensemble) : Ensemble des indices des k plus proches voisins.
# Retour :
# - n_set (ensemble) : Ensemble des indices des k plus proches voisins."""
def get_k_neighbors(voisins_directs, profondeur_k, point_index, n_set):
    if profondeur_k == 0:
        return n_set
    elif profondeur_k == 1:
        n_set.update(voisins_directs[point_index])
    else:
        for neighbor in voisins_directs[point_index]:
            n_set.update(get_k_neighbors(voisins_directs, profondeur_k - 1, neighbor, n_set))
    return n_set





    
def isthere_duplicat(mapping) :
    n = len(mapping)
    inv = dict()
    for k,v in mapping.items() :
        if isinstance(v , tuple) :
            triangle_indices = [v[3], v[4], v[5]]
            triangle = [mapping[triangle_indices[0]], mapping[triangle_indices[1]], mapping[triangle_indices[2]]]
            cartesian_point = tuple(barycentric_to_cartesian(triangle, v[:3]))
            inv[cartesian_point]  = k
        else :
            inv[tuple(v)] = k
    if len(inv) == len(mapping) :
        print("ok")




def constrained_triangulation_2D(points, boundary):
    """
    Effectue une triangulation de Delaunay contrainte sur un ensemble de points 2D
    avec des contraintes de frontières.
    """
    delaunay = Delaunay(points)
    triangles = []
    for simplex in delaunay.simplices:
        triangle = points[simplex]
        polygon = Polygon(triangle)
        if polygon.within(boundary):
            triangles.append(triangle)
    return triangles


def find_base_domain_vertices (face , mapping , vertices_base_domain) :
    v = []
    for v_index in [face.a, face.b, face.c] :
        if isinstance(mapping[v_index] , tuple) :
            alpha, beta, gamma, i, j, k = mapping[v_index]
            vertex = alpha * mapping[i] + beta *mapping[j] + gamma * mapping[k]
            v.append(vertex)
            vertices_base_domain[v_index] =vertex
        else :
            vertices_base_domain[v_index] = mapping[v_index]
            v.append(mapping[v_index])
    return v 
