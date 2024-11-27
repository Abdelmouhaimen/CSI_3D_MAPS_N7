"""
def get_tangent_vectors(self,n):
    if abs(n[0]) > abs(n[1]):
        u = np.array([-n[2], 0, n[0]]) / np.sqrt(n[0]**2 + n[2]**2)
    else:
        u = np.array([0, n[2], -n[1]]) / np.sqrt(n[1]**2 + n[2]**2)
    v = np.cross(n, u)
    return u, v"""
''' cCalculer la courbure en un point en estimant la surface par la méthode des moindres carrés
def calculate_curvature(self, vertex_index) :

    ni = self.compute_normal_vertex( vertex_index)
    u, v = self.get_tangent_vectors(ni)
    
    # Préparer les matrices pour la minimisation des moindres carrés
    A = []
    b = []
    neighbor_vertex = self.neighbor_vertices[vertex_index]

    for j in neighbor_vertex:
        Pj = self.vertices[j].T
        Pi = self.vertices[vertex_index].T
        P_diff = Pj - Pi
        uj = np.dot(P_diff, u)
        vj = np.dot(P_diff, v)
        wj = np.dot(P_diff, ni)
        
        A.append([uj**2, 2*uj*vj, vj**2])
        b.append(wj)
    
    A = np.array(A)
    b = np.array(b)
    
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    a, b, c = x

    H = a + c  
    K = a * c - b**2  

    discriminant = np.sqrt(H**2 - K)
    k1 = H + discriminant
    k2 = H - discriminant
    k =abs (k1) + abs(k2)

    return k'''

"""
Applique une carte conforme pour projeter les sommets voisins dans un plan.

def remove_vertex_and_retriangulate(self, vertex_index) :
    cycle = self.find_cyclique(self,vertex_index)
    neighbors = self.vertices[cycle,:]
    diff = neighbors - self.vertices[vertex_index]
    r  = np.linalg.norm(diff , axis= 1, keepdims=True)
    produits_scalaires = np.sum(diff[:-1 ,:] * diff[1: ,:], axis=1)
    cos_theta = produits_scalaires / (np.linalg.norm(diff[1:,:] ,axis= 1 )) * (np.linalg.norm(diff[:-1,:] ,axis= 1 ))
    theta = np.expand_dims(np.arccos(np.clip(cos_theta ,-1,1)),axis=1)
    upper = np.tril(np.ones((theta.shape[0],theta.shape[0])))
    total_angle = upper @ theta 
    positions_2D = np.zeros((len(self.neighbor_vertices[vertex_index]), 2))
    a = 2* np.pi /total_angle[-1]
    positions_2D[:,0] = (r**a) * np.cos(total_angle * a)  
    positions_2D[:,1] = (r**a) * np.sin(total_angle * a) 
    # Define the polygon (constraints boundary) 
    boundary = Polygon(positions_2D)
    # Create the triangulation within the boundary
    triangles = triangulate(boundary) 
    new_faces = []  
    for triangle in triangles: 
        points   = list(triangle.exterior.coords)[0:-1]
        face_vertex = []
        for pt in points :
            l = positions_2D == pt 
            l =  np.where((positions_2D == pt).all(axis=1))[0][0]
            ind =  cycle[l]
            face_vertex.append(ind)

        new_faces.append(Face.from_array_of_numbers(face_vertex))
    return new_faces """

"""
def conformal_map_projection_ofpoint(self, faces ,coordinates,vertex_index) :
    # Calcul de la projection conforme de p_i
    target_diff = target - center
    target_r = np.linalg.norm(target_diff)
    target_raw_theta = np.arctan2(target_diff[1], target_diff[0])  # Angle brut
    target_theta = target_raw_theta * scale_factor  # Applique le facteur d'échelle
    target_2D = np.array([target_r * np.cos(target_theta), target_r * np.sin(target_theta)])"""
"""
import numpy as np

def point_in_triangle(p, a, b, c):

    Vérifie si le point `p` est dans le triangle défini par `a`, `b`, `c`.
    Utilise les coordonnées barycentriques.
    
    # Vecteurs
    v0 = c - a
    v1 = b - a
    v2 = p - a

    # Coordonnées barycentriques
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)

def brown_faigle_point_location(q, triangles, vertices):

    # Choisir un triangle de départ arbitraire
    current_triangle_idx = 0
    visited = set()

    while True:
        # Récupérer le triangle courant
        visited.add(current_triangle_idx)
        triangle = triangles[current_triangle_idx]
        a, b, c = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

        # Vérifier si `q` est dans ce triangle
        if point_in_triangle(q, a, b, c):
            return current_triangle_idx  # Triangle trouvé

        # Heuristique pour choisir le prochain triangle
        for neighbor_idx in get_neighbors(current_triangle_idx, triangles):
            if neighbor_idx not in visited:
                # Tester la direction : si `q` est plus proche de ce voisin
                neighbor_triangle = triangles[neighbor_idx]
                neighbor_vertices = [vertices[i] for i in neighbor_triangle]
                if is_point_closer_to_triangle(q, neighbor_vertices):
                    current_triangle_idx = neighbor_idx
                    break
        else:
            raise ValueError("Point not found in triangulation. Verify input.")

def is_point_closer_to_triangle(q, vertices):

    centroid = np.mean(vertices, axis=0)  # Calcul du barycentre
    return np.linalg.norm(q - centroid)

def get_neighbors(triangle_idx, triangles):

    # Exemple fictif : liste de voisinage codée manuellement
    neighbors = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3],
        3: [1, 2]
    }
    return neighbors.get(triangle_idx, [])"""





        







"""
    def plot_conformal_map_projection(self, vertex_index):
        
        Plots the 2D conformal map projection of the 1-ring neighborhood of the specified vertex.
        
        :param vertex_index: Index of the vertex to be removed and whose 1-ring neighborhood is to be projected.
        
        new_faces, positions_2D, _ = self.remove_vertex_and_retriangulate(vertex_index)

        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')

        # Plot each triangle in the conformal mapping
        for triangle in positions_2D:
            polygon = Polygon(triangle)
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, edgecolor='k', facecolor='skyblue')

        # Plot the 2D positions of the vertices in the 1-ring
        cycle, _ = self.find_cyclique(vertex_index)
        cycle, isboundary = self.find_cyclique(vertex_index)
        if not isboundary:
            
            cycle.pop(0)
        else :
            cycle.pop(-1)
        print("nombre de triangle" , len(positions_2D))
        ndex_triangle = find_point_in_triangle(positions_2D , np.array([0,0]))
        print("triangle" ,new_faces[ndex_triangle] )
       #print(self.test_manifold(new_faces , cycle))
        for i, points in enumerate(positions_2D):
            for j,point in enumerate(points ):
                ax.plot(point[0], point[1], 'ro')  # Plot each vertex in red
                ax.text(point[0], point[1], f'{new_faces[i][j]}', fontsize=12, ha='right')  # Label vertices

        # Mark the origin (the vertex to be removed)
        ax.plot(0, 0, 'bo', markersize=8)  # Mark the vertex to be removed in blue
        ax.text(0, 0, f'{vertex_index}', fontsize=12, ha='right', color='blue')  # Label the removed vertex

        ax.set_title(f"Conformal Map Projection of 1-ring for Vertex {vertex_index}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.show()


    


    def plot_conformal_map_projection_with_edges_before_triangulation(self, vertex_index):
        
        Plots the 2D conformal map projection of the 1-ring neighborhood of the specified vertex,
        including edges between consecutive vertices, before triangulation.
        
        :param vertex_index: Index of the vertex whose 1-ring neighborhood is to be projected.
        
        # Get the positions of the 1-ring vertices in 2D conformal mapping
        positions_2D, a = self.conformal_mapping(vertex_index)
        #test
        #path =self.find_path(vertex_index)
        
        # Find the cyclic neighbors for labeling
        cycle, isboundary = self.find_cyclique(vertex_index)
        if not isboundary:
            
            cycle.pop(0)
        else :
            cycle.pop(-1)
        
        # Initialize the plot
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        #test
        #positions_2D = straighten_along_xaxis(cycle , path, positions_2D )
        # Plot each vertex in the conformal mapping and label it
        for i, point in enumerate(positions_2D):
            ax.plot(point[0], point[1], 'ro')  # Plot each vertex in red
            ax.text(point[0], point[1], f'{cycle[i]}', fontsize=12, ha='right')  # Label each vertex

        # Mark the origin (centered vertex to be removed)
        ax.plot(0, 0, 'bo', markersize=8)  # Mark the central vertex in blue
        ax.text(0, 0, f'{vertex_index}', fontsize=12, ha='right', color='blue')  # Label the central vertex
        
        # Draw edges between consecutive vertices in the 1-ring neighborhood
        for i in range(len(positions_2D)):
            start_point = positions_2D[i]
            end_point = positions_2D[(i + 1) % len(positions_2D)]  # Wrap around to form a loop
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 'g--')  # Green dashed line for edges

        # Configure the plot appearance
        ax.set_title(f"Conformal Map Projection (Before Triangulation) of 1-ring for Vertex {vertex_index}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.grid(True)
        plt.show()



    def plot_vertex_and_neighbors_with_edges(self, vertex_index):
        np.seterr(invalid = 'raise')
        
        vertices_3d = {i : list(self.neighbor_vertices[i]) for i in self.neighbor_vertices[vertex_index] } # Vous pouvez ajouter d'autres sommets ici si nécessaire
        vertices_3d[vertex_index] = list(self.neighbor_vertices[vertex_index])
        
            # Création de la figure 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Tracer chaque sommet
        for vertex, coord in vertices_3d.items():
            ax.scatter(coord[0], coord[1], coord[2], label=f'Vertex {vertex}')
            ax.text(coord[0], coord[1], coord[2], f'{vertex}', size=8)

        # Tracer les arêtes entre chaque sommet et ses voisins
        for vertex, neighbors in vertices_3d.items():
            for neighbor in neighbors :
                if neighbor in vertices_3d:  # Vérifier que le voisin a des coordonnées
                    x_values = [vertices_3d[vertex][0], vertices_3d[neighbor][0]]
                    y_values = [vertices_3d[vertex][1], vertices_3d[neighbor][1]]
                    z_values = [vertices_3d[vertex][2], vertices_3d[neighbor][2]]
                    ax.plot(x_values, y_values, z_values, 'b--')  # Ligne bleue en pointillés pour les arêtes

        # Configuration de l'affichage
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Graphe des sommets et voisins 1-ring de vertex 743")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.show()"""

