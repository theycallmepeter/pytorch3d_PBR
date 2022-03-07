
import torch
from torch.nn.functional import normalize

def tbn_matrices(mesh):
    """
    Calculates TBN matrices for the use in fragment shader, made following 
    https://learnopengl.com/Advanced-Lighting/Normal-Mapping as a guide

    Args: 
        mesh: pytorch3d Mesh object for which you want to calculate the matrices 
    """
    device = mesh.device()

    # Vertex coordinates in 3D
    verts = mesh.verts_packed()


    # Faces with indices of vertex
    faces = mesh.faces_packed()


    # Actual face coordinates in 3D
    faces_coords_3d =verts[faces]


    # Vertex coordinates in UV
    verts_uvs = mesh.textures.verts_uvs_padded().squeeze()


    # Faces with indices for vertex in UV 
    faces_uvs = mesh.textures.faces_uvs_padded().squeeze()

    # Actual face coordinates in UV
    faces_coords_uv = verts_uvs[faces_uvs]

    # Calculate edge vectors    
    edge1 = (faces_coords_3d[:, 1, :] - faces_coords_3d[:, 0, :])
    edge2 = (faces_coords_3d[:, 2, :] - faces_coords_3d[:, 0, :])
    deltauv1 = (faces_coords_uv[:, 1, :] - faces_coords_uv[:, 0, :])
    deltauv2 = (faces_coords_uv[:, 2, :] - faces_coords_uv[:, 0, :])

    # Precompute coefficient (not actually determinant, but called it that because it looks similar)    
    det = 1 / (deltauv1[:, 0] * deltauv2[:, 1] - deltauv1[:, 1] * deltauv2[:, 0])

    # Calculate tangent and bitangent vectors
    tangents = det.unsqueeze(1) * (deltauv2[:,1].unsqueeze(1) * edge1 - deltauv1[:, 1].unsqueeze(1) * edge2)
    bitangents = det.unsqueeze(1) * (-deltauv2[:,0].unsqueeze(1) * edge1 - deltauv1[:, 0].unsqueeze(1) * edge2)


    # Face normals (can also calculate them with cross product of the previous two vectors)
    face_normals = mesh.faces_normals_packed()

    # face_normals = torch.cross(tangents, bitangents, dim=1)
    
    # Normalize all vectors with L2 norm
    tangents = normalize(tangents, p=2, dim=1)
    bitangents = normalize(bitangents, p=2, dim=1)
    face_normals = normalize(face_normals, p=2, dim=1)


    # create matrices by stacking vectors along dimension, add identity matrix at end for fragments without any face 
    # (not sure if it does anything anyways as there is no light to be calculated anyways)
    TBN_matrices = torch.stack([tangents, bitangents, face_normals], dim=1)
    TBN_matrices = torch.cat([TBN_matrices, torch.eye(TBN_matrices.shape[1]).unsqueeze(0).to(device)])

    # Get inverse for light calculations in vertex shader, from https://stackoverflow.com/questions/5255806/how-to-calculate-tangent-and-binormal
    # T' = T - (N·T) N
    # U' = U - (N·U) N - (T'·U) T'
    # tangents_inv = tangents - (torch.tensordot(face_normals, tangents))*face_normals
    # bitangents_inv = bitangents - (torch.tensordot(face_normals, bitangents))*face_normals - (torch.tensordot(tangents_inv, bitangents))*tangents_inv
    # tangents_inv = normalize(tangents_inv, p=2, dim=1)
    # bitangents_inv = normalize(bitangents_inv, p=2, dim=1)

    # invTBN_matrices = torch.stack([tangents_inv, bitangents_inv, face_normals], dim=1)
    # invTBN_matrices = torch.cat([invTBN_matrices, torch.eye(invTBN_matrices.shape[1]).unsqueeze(0).to(device)])

