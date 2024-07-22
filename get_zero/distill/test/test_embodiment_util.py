import unittest
import torch
import os
import yaml
import numpy as np

from get_zero.distill.utils.embodiment_util import EmbodimentProperties, get_adjacency_matrix_from_urdf, compute_shortest_path_matrix, compute_shortest_path_edge_matrix, compute_spd_matrix

link_name_to_id_str = """
palm_lower: 0
mcp_joint: 1
pip: 2
dip: 3
fingertip: 4
pip_4: 5
thumb_pip: 6
thumb_dip: 7
thumb_fingertip: 8
mcp_joint_2: 1
pip_2: 2
dip_2: 3
fingertip_2: 4
mcp_joint_3: 1
pip_3: 2
dip_3: 3
fingertip_3: 4
"""

class TestEmbodimentUtil(unittest.TestCase):
    def test_get_adjacency_matrix_from_urdf(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(dir_path, 'assets/001.urdf')
        with open(urdf_path, 'r') as f:
            urdf = f.read()
        adjacency_matrix, joint_name_to_joint_i = get_adjacency_matrix_from_urdf(urdf)
        self.assertTrue(torch.all(adjacency_matrix == adjacency_matrix.T))

    def load_embodiment(self, provide_no_joint_name_to_joint_i=False):
        if provide_no_joint_name_to_joint_i:
            joint_name_to_joint_i = None
        else:
            joint_name_to_joint_i = {'0': 1, '1': 0, '10': 14, '11': 15, '12': 4, '13': 5, '14': 6, '15': 7, '2': 2, '3': 3, '4': 9, '5': 8, '6': 10, '7': 11, '8': 13, '9': 12}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(dir_path, 'assets/001.urdf')
        with open(urdf_path, 'r') as f:
            urdf = f.read()
        return EmbodimentProperties('001', urdf, joint_name_to_joint_i)

    def test_adjacency_matrix_to_spd(self):
        """Valides that the shortest path distance (SPD) matrix can be properly computed from the adjacency matrix. This is used for the Graphormer spatial positional encoding. This test also tests padding correctly occurs when provided adjacency matrix is larger than specified node count."""
        # order joints in same order as their names for more easy understanding
        # see top of assets/001.urdf for connectivity diagram
        embodiment_properties = self.load_embodiment(provide_no_joint_name_to_joint_i=True)
        A = embodiment_properties.adjacency_matrix.cpu().numpy()
        node_count = A.shape[0]
        # Add 1 row and col of padding
        A_padded = np.zeros((node_count+1, node_count+1))
        A_padded[:node_count, :node_count] = A
        SPD = compute_spd_matrix(A_padded, node_count)

        SPD_expected = np.array( # only first for rows for ease of computation
            [[0,1,1,2,3,2,4,5,3,2,4,5,2,3,4,5,-1],
             [1,0,2,3,2,1,3,4,2,1,3,4,1,2,3,4,-1],
             [1,2,0,1,4,3,5,6,4,3,5,6,3,4,5,6,-1],
             [2,3,1,0,5,4,6,7,5,4,6,7,4,5,6,7,-1],
             ]
        )
        self.assertTrue(np.all(SPD[:SPD_expected.shape[0]] == SPD_expected))
        self.assertEqual(SPD[-1, -1], -1)

    def test_shortest_path_matrix(self):
        print()
        embodiment_properties = self.load_embodiment()
        joint_name_to_joint_i = embodiment_properties.joint_name_to_joint_i
        adjacency_matrix = embodiment_properties.adjacency_matrix

        result = compute_shortest_path_matrix(adjacency_matrix.numpy())

        # check path from '2' to '11'
        # names: '2' -> '0' -> '1' -> '9' -> '8' -> '10' -> '11'
        expected_path = ['2', '0', '1', '9', '8', '10', '11']
        expected_path = [joint_name_to_joint_i[joint_name] for joint_name in expected_path]
        expected_path = expected_path + [-1] * (len(joint_name_to_joint_i) - len(expected_path))
        expected_path = np.array(expected_path, dtype=np.int32)
        i, j = joint_name_to_joint_i['2'], joint_name_to_joint_i['11']

        self.assertTrue(np.all(result[i,j] == expected_path))

        # check path from '11' to '2' is same path, but in reverse order
        expected_path = [x for x in expected_path if x != -1]
        expected_path.reverse()
        expected_path = expected_path + [-1] * (len(joint_name_to_joint_i) - len(expected_path))
        self.assertTrue(np.all(result[j,i] == expected_path))
    
    def test_compute_shortest_path_edge_matrix(self):
        # check edge matrix
        expected = torch.tensor([   [-2,  1, -1, -1,  0, -1, -1, -1,  0, -1, -1, -1,  0, -1, -1, -1],
                                    [ 1, -2,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1,  2, -2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1,  3, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [ 0, -1, -1, -1, -2,  5, -1, -1,  0, -1, -1, -1,  0, -1, -1, -1],
                                    [-1, -1, -1, -1,  5, -2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1,  6, -2,  7, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1,  7, -2, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [ 0, -1, -1, -1,  0, -1, -1, -1, -2,  1, -1, -1,  0, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1,  1, -2,  2, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1,  2, -2,  3, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -2, -1, -1, -1, -1],
                                    [ 0, -1, -1, -1,  0, -1, -1, -1,  0, -1, -1, -1, -2,  1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -2,  2, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2, -2,  3],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -2]], dtype=torch.int)

        link_name_to_id = yaml.safe_load(link_name_to_id_str)
        embodiment_properties = self.load_embodiment()
        joint_name_to_joint_i = embodiment_properties.joint_name_to_joint_i
        E = embodiment_properties.compute_edge_matrix(link_name_to_id)

        self.assertTrue(torch.all(expected == E))

        # check shortest_path_edge_matrix

        # '2' to '15'
        A = embodiment_properties.adjacency_matrix.numpy()
        SPM = compute_shortest_path_edge_matrix(A, E.numpy())
        i, j = joint_name_to_joint_i['2'], joint_name_to_joint_i['15']
        expected = ['pip', 'mcp_joint', 'palm_lower', 'pip_4', 'thumb_pip', 'thumb_dip']
        expected = [link_name_to_id[x] for x in expected]
        expected = expected + [-1] * (A.shape[0] - len(expected))
        self.assertTrue(np.all(SPM[i,j] == expected))

        # '15' to '2'
        expected = [x for x in expected if x != -1]
        expected.reverse()
        expected = expected + [-1] * (len(joint_name_to_joint_i) - len(expected))
        self.assertTrue(np.all(SPM[j,i] == expected))

        # '9' to '9' should be all -1
        expected = [-1] * A.shape[0]
        actual = SPM[9,9]
        self.assertTrue(np.all(actual == expected))

        # '13' to '14'
        expected = [link_name_to_id['thumb_pip']] + ([-1] * (A.shape[0] - 1))
        actual = SPM[joint_name_to_joint_i['13'], joint_name_to_joint_i['14']]
        self.assertTrue(np.all(actual == expected))

    def test_parent_child_matrix(self):
        expected_child = torch.tensor( [[0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        
        expected_parent = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0]], dtype=torch.int32)

        joint_name_to_joint_i = {'0': 1, '1': 0, '10': 14, '11': 15, '12': 4, '13': 5, '14': 6, '15': 7, '2': 2, '3': 3, '4': 9, '5': 8, '6': 10, '7': 11, '8': 13, '9': 12}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(dir_path, 'assets/001.urdf')
        with open(urdf_path, 'r') as f:
            urdf = f.read()
        embodiment_properties = EmbodimentProperties('001', urdf, joint_name_to_joint_i)

        actual_parent, actual_child = embodiment_properties.compute_parent_and_child_matrix()

        self.assertTrue(torch.all(actual_child == expected_child))
        self.assertTrue(torch.all(actual_parent == expected_parent))

if __name__ == '__main__':
    unittest.main()
