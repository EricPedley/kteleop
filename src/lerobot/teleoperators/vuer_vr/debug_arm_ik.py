from lerobot.teleoperators.vuer_vr.ik import KBot_ArmIK
import numpy as np

hand2inspire = np.array([[0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [1, 0, 0, 0],
                         [0, 0, 0, 1]])

arm_ik = KBot_ArmIK()

head_mat = np.array([[ 0.96218, -0.02616,  0.27115,  0.     ],
       [ 0.03057,  0.99946, -0.01207,  0.     ],
       [-0.27068,  0.01991,  0.96246,  0.     ],
       [-0.02576, -0.00401,  1.51409,  1.     ]])

right_wrist_mat = np.array([[ 0.73398,  0.18386, -0.65381, -0.0567 ],
       [-0.46233,  0.84045, -0.28267, -0.32159],
       [ 0.49752,  0.50975,  0.70188,  1.26799],
       [ 0.     ,  0.     ,  0.     ,  1.     ]])

rel_left_wrist_mat = np.array([[ 0., -1.,  0.,  0.],
       [ 0.,  0., -1.,  0.],
       [ 1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.]
])

left_wrist_mat = np.array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])

rel_left_wrist_mat = np.array([[-0.36466, -0.85663,  0.36497,  0.05492],
       [-0.48936, -0.15716, -0.8578 , -0.1955 ],
       [ 0.79218, -0.49141, -0.3619 ,  1.31465],
       [ 0.     ,  0.     ,  0.     ,  1.     ]])


rel_left_wrist_mat = left_wrist_mat @ hand2inspire
rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]

rel_right_wrist_mat = right_wrist_mat @ hand2inspire  # wTr = wTh @ hTr
rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]
       

# when left fingers is all zeros (homogenous coords) and rel_left_wrist_mat is identity, we need to not send the position.
joints = arm_ik.solve_ik(rel_left_wrist_mat, rel_right_wrist_mat)