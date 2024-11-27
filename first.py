import obja
import numpy as np
import sys


import shapely
from shapely.geometry import Polygon, MultiPoint, LineString
from shapely.ops import triangulate
from obja import Face
from utilites import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm





class Decimater(obja.Model):
    """
    
    """
    def __init__(self):
        super().__init__() 
        self.deleted_faces =  set()
        self.neighbor_faces = {}
        self.neighbor_vertices = {}
        self.deleted_vertices =  []
        self.tagged_edges = set()
        self.dihedral_angle_threshold = np.deg2rad(0)
        self.curvature_threshold = float('inf')
        self.special_vertices = set()
        self.new_faces = []
        # Initialise les mappings du niveau initial
        self.old_deleted_vertices = []
        self.boundary_edges = set()
        self.cycles_edges = set()
        self.operations = []
        self.faces_all_levels = []
        self.map_faces_to_facesAll = {}
        
      

    '''Trouver les faces voisines de tous les sommets et les stocker dans un dictionnaire'''
    def find_1ring_neighborhood(self) :
        
        for  (vertex_index, vertex) in enumerate(self.vertices):
            if vertex_index not in self.old_deleted_vertices :
                self.neighbor_faces [vertex_index] =  []
                self.neighbor_vertices[vertex_index] = set()
                # Create a dictionary to store edge-face incidences
                edge_face_incidences = {}
                for (face_index, face) in enumerate(self.faces):
                    face_vertices = [face.a,face.b,face.c]
                    if  vertex_index in  face_vertices :
                        # Create edges for the face
                        for i in range(3):
                            edge = tuple(sorted([face_vertices[i], face_vertices[(i + 1) % 3]]))
                            if edge not in edge_face_incidences:
                                edge_face_incidences[edge] = []
                            edge_face_incidences[edge].append(face_index)


                        self.neighbor_faces[vertex_index].append(face_index)
                        # Trouver les indices des sommets voisins dans le 1-ring. 
                        face_vertices.remove(vertex_index)
                        self.neighbor_vertices[vertex_index].update(face_vertices)

                # Check for non-manifold edges
                for edge, faces in edge_face_incidences.items():
                    if len(faces) > 2:
                        print("le maillage est non manifold")
                        sys.exit()
                        
                
    def find_boundary_edges(self):
        """
        Trouve toutes les arêtes de frontière dans le maillage.
        :return: Un ensemble d'arêtes de frontière.
        """
        edges = dict()  # Un dictionnaire pour suivre les arêtes partagées

        # Parcours des faces du maillage
        for face_index, face in enumerate(self.faces):
            # Définir les arêtes de la face (chaque arête est un tuple ordonné de deux indices de sommets)
            face_vertices = [face.a, face.b, face.c]
            for i in range(3):
                edge = tuple(sorted([face_vertices[i], face_vertices[(i + 1) % 3]]))  # Ordonner les indices pour éviter les doublons
                if edge in edges:
                    edges.pop(edge)  # Si l'arête est partagée, on la retire
                else:
                    edges[edge] = 1  # Sinon, on l'ajoute comme arête non partagée (frontière)

        # Les arêtes restantes dans le dictionnaire sont celles qui ne sont partagées que par une seule face
        self.boundary_edges.update(edges.keys())
        

            


    def detect_tagged_feature_lines(self) :
        """
        Détecter les aretes caractéristiques  basées sur les angles dièdres.
        Une arete est marquée comme caractéristique si l'angle entre ses faces adjacentes
        est inférieur à un certain seuil.
        """
        for face in self.faces :
            face_vertices = [face.a, face.b, face.c]
            for i in range(3):
                edge =  tuple(sorted([face_vertices[i],face_vertices[(i+1)%3]]))
                if edge in self.tagged_edges:
                    continue  
                angle = self.calcule_dihedral_angle(edge) 
                if angle : 
                    if angle < self.dihedral_angle_threshold:
                        self.tagged_edges.add(edge)
            



    def calcule_dihedral_angle(self , edge) :
        """ Calculer l'angle dièdre  d une arrete"""
        adjacent_faces = [f for f in self.neighbor_faces[edge[0]] if edge[1] in [self.faces[f].a , self.faces[f].b , self.faces[f].c ]]
        if len(adjacent_faces) == 2:
            # Calculer l'angle dièdre entre les normales des deux faces adjacentes
            face1, face2 = adjacent_faces
            normal1, _ = self.normal_surface(face1)
            normal2, _ = self.normal_surface(face2)
            cos_theta = np.clip(np.dot(normal1, normal2), -1, 1)
            return np.arccos(cos_theta)
        else : 
            return None


    def compute_normal_vertex(self, vertex_index):
        '''Calculer la normale en chaque point de notre maillage 3D 
        comme la moyenne pondérée des normales des surfaces voisines
        '''
        neighbor_faces  = self.neighbor_faces[vertex_index]
        n = np.zeros(3)
        A = len(neighbor_faces)
        for face_index in neighbor_faces :
            ni,ai =self.normal_surface(face_index )
            n += ni/A
            
        return n/np.linalg.norm(n)
    
    
    def normal_surface(self ,face_index) :
        ''' Calculer la normale de chaque face voisine de notre point '''
        face = self.faces[face_index]
        v1 = self.vertices[face.b] - self.vertices[face.a]
        v2 = self.vertices[face.c] - self.vertices[face.a]
        ni = np.cross(v1,v2)
        ai = np.linalg.norm(ni)
        return ni/ai ,ai
    

    
    def calculate_curvature(self, vertex_index) :
        origin = self.vertices[vertex_index]
        knn = list(compute_knn(2, self.neighbor_vertices,vertex_index))
        n_components =3
        vertices = np.array(self.vertices)
        data = vertices[knn,:]
        if len(knn) > 6 :
            new_coord = make_pca(data=data , origin=origin , n_components= n_components)
            X  = lstsq_quadrics_fitting(new_coord)
            a, b, c,d,e,_ = X.ravel()
            x,y,_ = self.vertices[vertex_index].ravel()
            """
            # Calcul de la courbure moyenne H et de la courbure gaussienne K
            # Construire la matrice Hessienne
            H = np.array([[2*a , c] , [c,2*b]])
            # Calculer les valeurs propres (courbures principales)
            eigenvalues, _ = np.linalg.eig(H)
            k1,k2 = eigenvalues.ravel()"""
            # Calcul des dérivées premières et secondes
            f_x = 2*a*x + c*y + d
            f_y = 2*b*y + c*x + e
            f_xx = 2*a
            f_yy = 2*b
            f_xy = c

            # Première forme fondamentale
            E = 1 + f_x**2
            F = f_x * f_y
            G = 1 + f_y**2
            I = np.array([
                [E, F],
                [F, G]
            ])

            # Deuxième forme fondamentale
            II = np.array([
                [f_xx, f_xy],
                [f_xy, f_yy]
            ])

            # Inverser la première forme fondamentale
            I_inv = np.linalg.inv(I)

            # Calcul de la matrice des courbures
            S = I_inv @ II

            # Calcul des valeurs propres
            k1, k2 = np.linalg.eigvalsh(S)
            K =  abs(k1)  + abs(k2)
            return K
        return 0


    

    '''for a vertex p_i ∈P_l, we consider its 1-ring neighborhood φ(|star(i)|) and compute its area a(i)   '''
    def calculate_area(self , vertex_index):
        area = 0
        for face_index in self.neighbor_faces[vertex_index] :
            _,ai = self.normal_surface(face_index)
            area += ai/2
        return area 
    

    def calculate_priority(self,lambd =1/2 ) :
        unremovable = set()
        A = []
        K = []
        max_A = -1
        max_K = -1
        for  (vertex_index, vertex) in enumerate(self.vertices):
            if vertex_index not in self.old_deleted_vertices:
                ai = self.calculate_area(vertex_index)
                ki = self.calculate_curvature(vertex_index)
                if ki > self.curvature_threshold  or self.is_dart_vertex(vertex_index)[0] or  self.is_dart_vertex(vertex_index)[1] >2 :
                    
                    self.special_vertices.add(vertex_index)
            else : 
                ai = -1
                ki = -1
            A.append(ai)
            K.append(ki)

            max_A, max_K = max(max_A, ai), max(max_K, ki)

        w = list((lambd/max_A) * np.array(A)  + ((1 - lambd)/ max_K)* np.array(K))
        dict = {i:l for i,l in enumerate(w)}
        indices = sorted(dict  , key= lambda x : dict[x])
        for i in indices:
            if i not in self.old_deleted_vertices :
                if len(list(self.neighbor_vertices[i])) < 12 and i not in unremovable and i not in self.special_vertices :
                        self.deleted_vertices.append(i)
                        unremovable.update(self.neighbor_vertices[i])
                        self.deleted_faces.update(self.neighbor_faces[i])
                   
    

    def find_cyclique(self,vertex_index) :
        """
        Organise les sommets voisins d'un sommet donné dans un ordre cyclique
        en suivant les arêtes dans le maillage.
        """
        isboundary , edges = self.is_boundary_vertex(vertex_index)
        ring_1  = list(self.neighbor_vertices[vertex_index])
        if isboundary  :
            edge = edges[0]
            current_vertex = edge[0] if vertex_index == edge[1] else edge[1]
        else :
            current_vertex =  ring_1[0]
        cyclique_ring_1 = []

        # Fonction récursive pour trouver le cycle
        def recursive_cycle(current_vertex, visited):
            """
            Fonction récursive qui explore les voisins de `current_vertex` pour construire le cycle.
            """
            # Ajouter le sommet actuel au cycle et le marquer comme visité
            visited.append(current_vertex)
            # Si on a visité tous les voisins du 1-ring et qu'on revient au départ, cycle complet
            if not isboundary :
                if len(visited) == len(ring_1) and visited[0] in self.neighbor_vertices[visited[-1]]:
                    return visited  
            else :
                if len(visited) == len(ring_1) :
                    return visited  
            # Explorer les voisins du sommet courant
            for neighbor in self.neighbor_vertices[current_vertex]:
                if neighbor not in visited and neighbor in ring_1:
                    result = recursive_cycle(neighbor, visited)
                    if result:  # Si le cycle est complet, retourner le résultat
                        return result
            # Si aucun cycle n'est trouvé, faire marche arrière (backtrack)
            visited.pop()
            return None
        # Démarrer la récursion depuis le sommet de départ
        cyclique_ring_1 = recursive_cycle(current_vertex, [])

        return cyclique_ring_1 + [current_vertex],isboundary
    
    

   
    def is_boundary_vertex(self, vertex_index) :
        """
        Vérifie si un sommet appartient à la frontière.
        :param vertex_index: Index du sommet à vérifier.
        :return: (bool, list) - True si le sommet est sur la frontière, et la liste des arêtes de frontière associées.
        """
        boundary_edges = []
        for v in self.neighbor_vertices[vertex_index] :
            edge = tuple(sorted([vertex_index , v]))
            if edge in self.boundary_edges :
                boundary_edges.append(edge)
        return len(boundary_edges) >0 , boundary_edges
    

    def is_dart_vertex(self,vertex_index) :
        i = 0
        for v in self.neighbor_vertices[vertex_index] :
            edge = tuple(sorted((vertex_index,v)))
            if edge in self.tagged_edges :
                i+=1
        return i == 1 ,i
    
    def find_path(self , vertex_index) :
        path = []
        for v in self.neighbor_vertices[vertex_index] :
            edge = tuple(sorted((vertex_index,v)))
            for e in self.tagged_edges :
                if e == edge :
                    path.append(v)
        return path

                

    
    def conformal_map_projection_ofpoint(self, faces ,coordinates,vertex_index , removed_vertex) :
        # Calcul de la projection conforme de p_i
        coords = []
        for idx  in self.mapping[vertex_index][3:] :
            if idx != removed_vertex : 
                for i,f in enumerate(faces) : 
                    l =  [f.a , f.b , f.c]
                    if idx in l :
                        coord = coordinates[i][l.index(idx)]
                        coords.append(coord)
                        break
            else : 
                coords.append(np.array([0,0]))
        target2D = np.array([0,0])
        for i in range(3) :
            target2D = target2D + self.mapping[vertex_index][i] * coords[i]
        return target2D


    """ calculer la map en utilisant le 1_anneau  """
    def  conformal_mapping(self, vertex_index ):
        cycle , isboundary = self.find_cyclique(vertex_index)
        
        np_vertices = np.array(self.vertices)
        neighbors = np_vertices[cycle,:]
        diff = neighbors - self.vertices[vertex_index]
        r  = np.linalg.norm(diff , axis= 1, keepdims=True)
        produits_scalaires = np.sum(diff[:-1 ,:] * diff[1: ,:], axis=1)
        cos_theta = produits_scalaires / (np.linalg.norm(diff[1:,:] ,axis= 1) * np.linalg.norm(diff[:-1,:] ,axis= 1))
        theta = np.expand_dims(np.arccos(np.clip(cos_theta ,-1,1)),axis=1)

        if isboundary :
            
            theta = np.vstack((np.array([0]), theta))
            theta = np.delete(theta , -1 , axis= 0)
            total_angle = np.cumsum(theta)
            # 'a' représente un facteur d'échelle des angles en fonction de l'angle total
            a =  np.pi /total_angle[-1]
            r = r[:-1,:]
        else :
            r = r[1:,:]
            total_angle = np.cumsum(theta)
            # 'a' représente un facteur d'échelle des angles en fonction de l'angle total
            a =  2 * np.pi /total_angle[-1]
        total_angle = total_angle * a       
        
        return (r**a) * np.column_stack((np.cos(total_angle), np.sin(total_angle))) , a
    



    
    """ supprimer un point et retrianguler son 1_anneau """
    def remove_vertex_and_retriangulate(self, vertex_index):
        
        positions_2D,a = self.conformal_mapping(vertex_index)
        
        cycle, isboundary  = self.find_cyclique(vertex_index)
        if not isboundary:
            cycle.pop(0)
        else :
            cycle.pop(-1)
        # cycle edges juste pour la verification 
        boundary_edges = [tuple(sorted([cycle[i], cycle[(i + 1) % len(cycle)]])) for i in range(len(cycle))]
        self.cycles_edges.update(boundary_edges)
        
        if  self.is_dart_vertex(vertex_index)[1] == 2 :
            
            path =self.find_path(vertex_index)
            positions_2D = straighten_along_xaxis(cycle , path, positions_2D )
            triangles = retrangulate_upper_lower_parts(path, cycle ,  positions_2D)
        else :
            tableau = np.vstack((positions_2D,positions_2D[0,:]))
            boundary = Polygon(tableau)
            triangles = triangulate(boundary) 
            triangles = [tri for tri in triangles if tri.within(boundary)]
 
            
        new_faces = []  
        new_traingles = []
        dictedges = dict()
        for triangle in triangles:
            
            points   = list(triangle.exterior.coords)[0:-1]
            #points = list(triangle)
            triangle = []
            face_vertices = []
            for p in points :
                face_vertices.append(cycle[np.argwhere((positions_2D == p).all(axis=1))[0][0]])
                triangle.append(np.array(p))
            ####################################################################################
            # Vérifiez les arêtes pour éviter les duplications excessives
            edges = [tuple(sorted((face_vertices[i], face_vertices[(i+1) % 3]))) for i in range(3)]
            for edge in edges:
                dictedges[edge]  = dictedges.get(edge,0) +1
                if edge in boundary_edges :
                    if dictedges[edge] > 1 :
                        print("ach had l khra")
                else :
                    if dictedges[edge] > 2 :
                        print("Erreur : arête non-manifold détectée", edge)
            ################################################################################
            new_traingles.append(triangle) 
            new_faces.append(Face.from_array_of_numbers(face_vertices))
            
            
        
        if not self.test_manifold(new_faces , cycle):
            print("l9it mok")
    
        return new_faces,new_traingles,a
    
    
    
    def parameterize(self):
        
        ###################################
        already_removed = set()
        for  (vertex_index, vertex) in enumerate(self.vertices):
            
                if vertex_index  in self.deleted_vertices  and vertex_index not in already_removed :

                    
                    new_faces, new_traingles ,a  = self.remove_vertex_and_retriangulate(vertex_index)
                    
                    index_triangle = find_point_in_triangle(new_traingles , np.array([0,0]))
                    
                    u,v,w,_ =  assign_barycentric_coords(triangle= new_traingles[index_triangle], point=np.array([0,0]))
                
                    face = new_faces[index_triangle]
                    self.mapping[vertex_index] =  (u,v,w,face.a,face.b,face.c)
                    
                    ###### a modifier apres ##########
                    self.new_faces.extend(new_faces)
                    index_first_face_next_level_next_vertice = len(self.faces_all_levels)
                    self.faces_all_levels.extend(new_faces)
                    #for face_index, face in enumerate(new_faces):
                        #self.map_faces_to_facesAll[face_index + index_first_face_next_level_next_vertice] = face_index

                    

                    #################################  Enregistrer les operations ####################################
                    # Enregistrer la suppression des faces affectées
                    for face_index in self.neighbor_faces[vertex_index]:
                        self.operations.append(('face', self.map_faces_to_facesAll[face_index], self.faces[face_index]))
                    # Enregistrer la suppression du sommet
                    self.operations.append(('vertex', vertex_index, self.vertices[vertex_index]))

                    for ind_in_new_faces,new_face in enumerate(new_faces):
                        self.operations.append(('delete_face', index_first_face_next_level_next_vertice + ind_in_new_faces, new_face))
                        

                else :
                    
                    if isinstance(self.mapping[vertex_index] , tuple)  :
                        
                        for v_index in self.mapping[vertex_index][3:] :
                            if v_index in  self.deleted_vertices :
                                
                                new_face, new_traingles ,a  = self.remove_vertex_and_retriangulate(v_index)
                                ######################################################################################################
                                target_2D =  self.conformal_map_projection_ofpoint(new_face , new_traingles , vertex_index , v_index)
                                index_triangle = find_point_in_triangle(new_traingles , target_2D)
                                u,v,w,_ =  assign_barycentric_coords(triangle= new_traingles[index_triangle], point=target_2D)
                                
                                face = new_face[index_triangle]
                                self.mapping[vertex_index] =  (u,v,w,face.a,face.b,face.c)
                            
    
    def contract(self , l ,output) :
        self.faces_all_levels = self.faces.copy()
        self.index_first_face_current_level = 0
        self.index_first_newface_next_level = len(self.faces_all_levels)
        for face_index, face in enumerate(self.faces):
            self.map_faces_to_facesAll[face_index] = face_index

        self.mapping = {k : v for  k,v in enumerate(self.vertices)}
        self.faces_original = self.faces.copy()
        for k in range(l) :
            print("niveau" ,k)
            self.find_1ring_neighborhood()
            self.detect_tagged_feature_lines()
            self.find_boundary_edges()
            self.calculate_priority()
            self.parameterize()
            self.old_deleted_vertices.extend(self.deleted_vertices)

            print ("deleted_faces" ,len(self.deleted_faces))
            print ("deleted_vertices" , len(self.deleted_vertices))
            print ("new faces" ,len(self.new_faces))

            new_i = 0
            for ind, value in enumerate(self.faces):
                if  ind not in self.deleted_faces:
                    self.map_faces_to_facesAll[new_i] = self.map_faces_to_facesAll[ind]
                    new_i+=1
            for face_index, face in enumerate(self.new_faces):
                self.map_faces_to_facesAll[new_i] = self.index_first_newface_next_level + face_index
                new_i+=1
            self.index_first_newface_next_level = len(self.faces_all_levels)

            self.faces = [value for i, value in enumerate(self.faces ) if i not in self.deleted_faces]
            self.faces.extend(self.new_faces)


            ###### Verfication ################
            edges_dict = {}
            for face_index , face in enumerate(self.faces):
                face_vertices = [face.a , face.b, face.c]
                for i in range(3) :
                    edge = tuple(sorted((face_vertices[i], face_vertices[(i+1) % 3])))
                    edges_dict[edge] = edges_dict.get(edge,0) +1
            ###########
            for edge in edges_dict :
                if edge in self.cycles_edges :
                    if edges_dict[edge] > 2 :
                        print("l mockkil f triangulation" , edge)
                else :
                    if edges_dict[edge] > 2 :
                        print(edge)
                        print("ma3rftch")

            self.clear()
            self.index_first_face_current_level = len(self.faces_all_levels)
        
        deleted_faces_obja = set()
        # Iterate through the vertex
        print("index_first_face_current_level:", self.index_first_face_current_level)
        print("map_faces_to_facesAll:", self.map_faces_to_facesAll)
        for vertex_index,vertex in enumerate(self.vertices):
            if  vertex_index not in self.old_deleted_vertices:
                for (face_index, face) in enumerate(self.faces):

                    # Delete any face related to this vertex
                    if face_index not in deleted_faces_obja:
                        if vertex_index in [face.a,face.b,face.c]:
                            deleted_faces_obja.add(face_index)
                            # Add the instruction to operations stack
                            self.operations.append(('face', self.map_faces_to_facesAll[face_index], face))
                
                # Delete the vertex
                self.operations.append(('vertex', vertex_index, vertex))

        # To rebuild the model, run operations in reverse order
        # To rebuild the model, run operations in reverse order
        self.operations.reverse()

        # Write the result in output file
        output_model = obja.Output(output, random_color=True)

        for (ty, index, value) in self.operations:
            if ty == "vertex":
                output_model.add_vertex(index, value)
            elif ty == "face":
                output_model.add_face(index, value)   
            else:
                output_model.delete_face(index)
                            

    def export_to_obj(self, filename):
        """
        Exporte les sommets et les faces du modèle dans un fichier .obj.
        
        :param filename: Le nom du fichier de sortie.
        """
        
        with open(filename, 'w') as f:
            # Exporter les sommets
            faces_tick = dict()
            i=1
            for vertex_index,vertex in enumerate(self.vertices):
                if vertex_index not in self.old_deleted_vertices :
                    faces_tick[vertex_index] = i
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
                    i+=1
            
            # Exporter les faces
            for face in self.faces:
                f.write(f"f {faces_tick[face.a]} {faces_tick[face.b]} {faces_tick[face.c]}\n")


    def clear(self) :
        self.deleted_faces.clear()
        self.neighbor_faces.clear()
        self.neighbor_vertices .clear()
        self.deleted_vertices.clear()
        self.tagged_edges.clear()
        self.new_faces.clear()
        self.boundary_edges.clear()
        ###########################
        self.cycles_edges.clear()
        


    def test_manifold(self, new_faces, cycle):
        
        edge_face_incidences = {}
        
        for face in new_faces:
            face_vertices = [face.a, face.b, face.c]
            for i in range(3):
                edge = tuple(sorted([face_vertices[i], face_vertices[(i + 1) % 3]]))
                if edge not in edge_face_incidences:
                    edge_face_incidences[edge] = []
                edge_face_incidences[edge].append(face)
                
                # Vérifie que chaque arête est partagée par exactement deux faces
                if len(edge_face_incidences[edge]) > 2:
                    print("Erreur : Arête non-manifold détectée pour l'arête", edge)
                    return False
        
        # Vérification que toutes les arêtes du cycle sont respectées
        for i in range(len(cycle)):
            edge = tuple(sorted([cycle[i], cycle[(i + 1) % len(cycle)]]))
            if edge not in edge_face_incidences:
                print("Erreur : arête de frontière manquante dans la triangulation")
                return False

        return True




    # Uniform Remeshing
    def find_triangle_base(self, point): # VERIFY THAT POINTS ARE STORED AS NP.ARRAY SO WE CAN DO OPERATIONS ON THEM
        """
        Finds the triangle from the original pesh to the given point.
        """
        # Vertexes of the base domain
        vertices_base_domain = dict()
        distances = []
        for face_index, face in enumerate(self.faces_original):
            # Calculate barycentric coordinates of face
            triangle = find_base_domain_vertices (face , self.mapping ,vertices_base_domain)
            triangle_center = np.mean(triangle, axis=0)
            distances.append(np.linalg.norm(triangle_center - point))
        min_index = np.argmin(distances)
        indexes = [self.faces_original[min_index].a, self.faces_original[min_index].b, self.faces_original[min_index].c]
        return [(i, vertices_base_domain[i]) for i in indexes]
    
        
    def Map_back_midpoint(self, midpoint):
        """
        Maps back the midpoint to the original domain.
        """
        triangle_points = self.find_triangle_base(midpoint)
        #verify that midpoint is different from the closest points and return the index of the closest point
        if triangle_points is None:
            return midpoint
        for vertex_index, point in triangle_points:
            if np.allclose(point, midpoint):
                return self.vertices[vertex_index]
        
        alpha, beta, gamma, _ = assign_barycentric_coords(triangle=[point for _, point in triangle_points], point=midpoint)
        a, b, c = [self.vertices[ind] for ind, _ in triangle_points]
        return alpha * a + beta * b + gamma * c


    def uniform_remeshing(self, m=1):
        """
        Performs uniform remeshing of the mesh by subdividing each face into four smaller faces.
        :param error_threshold: The maximum allowable error for approximation.
        :param m: The number of levels of subdivision to perform.
        """
        self.edge_midpoints= {}
        self.len_vertices_b4_subdivision = len(self.vertices)
        for _ in range(m):
            new_faces = []
            for face in self.faces:
                new_faces.extend(self.subdivide_face(face))
            self.faces = new_faces
        # Mapping back the edge_midpoints to the finest level
        for midpoint_idx in tqdm(self.edge_midpoints.values(), desc="Mapping back midpoints"):
            self.vertices[midpoint_idx] = self.Map_back_midpoint(self.vertices[midpoint_idx])
        self.export_to_obj("uniform_remeshed_output.obj")
        
    def compute_face_error(self, face):
        """
        Computes the maximum error for a face by comparing mapped vertices with their approximations.
        """
        if tuple(sorted([face.a, face.b, face.c])) not in self.face_to_vertexOrigin:
            return 0
        verticesIndexes_assigned_to_face = self.face_to_vertexOrigin[tuple(sorted([face.a, face.b, face.c]))]
        if len(verticesIndexes_assigned_to_face) == 0:
            return 0
        vertices = np.array([self.vertices[vertex_index] for vertex_index in verticesIndexes_assigned_to_face])
        face_barycenter = np.mean([self.vertices_mapped_back[vertex_index] for vertex_index in [face.a, face.b, face.c]], axis=0)
        distances = np.linalg.norm(vertices - face_barycenter, axis=1)
        return np.max(distances)


    def adaptive_remeshing(self, error_threshold=0.01):
        """
        Performs adaptive remeshing of the mesh based on a given error threshold.
        :param error_threshold: The maximum allowable relative error for approximation.
        """
        # Calculate Bounding Box
        min_x, min_y, min_z = np.min(self.vertices, axis=0)
        max_x, max_y, max_z = np.max(self.vertices, axis=0)
        B = max(max_x - min_x, max_y - min_y, max_z - min_z)

        self.edge_midpoints = {}
        self.len_vertices_b4_subdivision = len(self.vertices)
        self.face_to_vertexOrigin = {}
        self.vertices_mapped_back = self.vertices.copy()
        
        # Map each vertex from the original domain to a triangle in the base domain
        for vertex_index,mapping in self.mapping.items():
            if vertex_index not in self.old_deleted_vertices:
                continue
            else:
                _, _, _, i, j, k = mapping
                key = tuple(sorted((i,j,k)))
                if key not in self.face_to_vertexOrigin:
                    self.face_to_vertexOrigin[key] = [vertex_index]
                else:
                    self.face_to_vertexOrigin[key].append(vertex_index)
        

                


        # Initialize a list to hold new faces after subdivision
        new_faces = []

        # Initialize a queue with faces that need to be processed
        faces_to_process = [face for face in self.faces]

        
        # press key to stop the loop
        face_barycenter_min = np.array([0, 0, 0])
        while faces_to_process:
            face = faces_to_process.pop()

            
            max_error = self.compute_face_error(face)
            print(f"Face {face}: {max_error/B}")


            if tuple(sorted([face.a, face.b, face.c])) not in self.face_to_vertexOrigin:
                new_faces.append(face)
                continue
            if len(self.face_to_vertexOrigin[tuple(sorted([face.a, face.b, face.c]))]) == 1:
                if max_error/B > error_threshold:
                    subdivided_faces = self.subdivide_face(face)
                    for new_face in subdivided_faces:
                        new_faces.append(new_face)
                else:
                    new_faces.append(face)
                continue
            if max_error/B > error_threshold:
                # Subdivide the face
                subdivided_faces = self.subdivide_face(face)

                # Assign vertices to the new face
                faces_barycenters = []
                for new_face in subdivided_faces:
                    new_face_barycenter = np.mean([self.vertices_mapped_back[vertex_index] for vertex_index in [new_face.a, new_face.b, new_face.c]], axis=0)
                    print("subdivided_faces", new_face)
                    print("face vertices", [self.vertices[vertex_index] for vertex_index in [new_face.a, new_face.b, new_face.c]])
                    faces_barycenters.append(new_face_barycenter)
                    self.face_to_vertexOrigin[tuple(sorted([new_face.a, new_face.b, new_face.c]))] = []
                faces_barycenters = np.array(faces_barycenters)

                verticesIndexes_assigned_to_face = self.face_to_vertexOrigin[tuple(sorted([face.a, face.b, face.c]))]
                vertices = np.array([self.vertices[vertex_index] for vertex_index in verticesIndexes_assigned_to_face])
                print("vertices assigned to face",face, verticesIndexes_assigned_to_face)
                for vertex_index,vertice in enumerate(vertices):
                    distances = np.linalg.norm(faces_barycenters - vertice, axis=1)
                    min_index = np.argmin(distances)
                    new_face = subdivided_faces[min_index]
                    self.face_to_vertexOrigin[tuple(sorted([new_face.a, new_face.b, new_face.c]))].append(verticesIndexes_assigned_to_face[vertex_index])
                print("face barycenter min", faces_barycenters[min_index])





                # Add new faces to processing queue
                for new_face in subdivided_faces:
                    #new_faces.append(new_face)
                    # Add to processing queue
                    faces_to_process.append(new_face)

            else:
                # Keep the face as is
                new_faces.append(face)
            print("len faces to process", len(faces_to_process))
        
        

        self.faces = new_faces

        # Map midpoints back to the finest level
        for midpoint_idx in tqdm(self.edge_midpoints.values(), desc="Mapping back midpoints"):
            self.vertices[midpoint_idx] = self.Map_back_midpoint(self.vertices[midpoint_idx])

        self.export_to_obj("adaptive_remeshed_output.obj")

                    








                            
                    










    



    

    





   






        

        





        
        


                
            







        

        
        

        
        



                


            



            









    
    
                    








