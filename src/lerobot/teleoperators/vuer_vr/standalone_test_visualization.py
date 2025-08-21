import math
from asyncio import sleep
from pathlib import Path

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Urdf, Hands


from urdfpy import URDF
import io
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret

pi = 3.1415

file_absolute_parent = str(Path(__file__).absolute().parent)

def load_urdf_with_absolute_paths(urdf_relative_path, assets_folder):
    """Load URDF with absolute mesh paths"""
    urdf_path = f'{file_absolute_parent}/{urdf_relative_path}'
    urdf_file_contents = open(urdf_path).read()

    # Replace both package:/ paths and relative mesh paths with absolute paths
    urdf_assets_path = f'{file_absolute_parent}/{assets_folder}'
    urdf_file_contents = urdf_file_contents.replace('package:/', file_absolute_parent)
    # Replace specific patterns - order matters to avoid double replacement
    urdf_file_contents = urdf_file_contents.replace('filename="./meshes/', f'filename="{urdf_assets_path}/meshes/')
    urdf_file_contents = urdf_file_contents.replace('filename="meshes/', f'filename="{urdf_assets_path}/meshes/')

    class DumbHack(io.StringIO):
        name = 'dummy name to get urdfpy to comply'
    urdf_file = DumbHack(urdf_file_contents)

    return URDF.load(urdf_file)

# Load all three robots
right_hand_robot = load_urdf_with_absolute_paths('assets/inspire_hand/inspire_hand_right.urdf', 'assets/inspire_hand')
print(right_hand_robot.actuated_joint_names)
left_hand_robot = load_urdf_with_absolute_paths('assets/inspire_hand/inspire_hand_left.urdf', 'assets/inspire_hand')
kbot_robot = load_urdf_with_absolute_paths('assets/kbot/robot.urdf', 'assets/kbot')

app = Vuer(static_root=Path(__file__).parent / "assets")

head_matrix_shared = np.zeros((4, 4), dtype=np.float32)
left_hand_shared = np.zeros((4, 4), dtype=np.float32)
left_landmarks_shared = np.zeros((25,4), dtype=np.float32)

right_hand_shared = np.zeros((4, 4), dtype=np.float32)
right_landmarks_shared = np.zeros((25,4), dtype=np.float32)

vuer_to_urdf_mat = Rotation.from_euler('xz', (90, 90), degrees=True).as_matrix()

tip_indices = [4, 9, 14, 19, 24]
def right_hand_inverse_kinematics(right_hand_tip_poses):

    link_names = [
        'R_thumb_tip','R_index_tip', 'R_middle_tip', 'R_ring_tip', 'R_pinky_tip' 
    ]
    _joint_names = ['R_thumb_proximal_yaw_joint', 'R_index_proximal_joint', 'R_middle_proximal_joint', 'R_ring_proximal_joint', 'R_pinky_proximal_joint', 'R_thumb_proximal_pitch_joint']

    def residuals(joint_angle_vector):
        '''corresponds to '''
        cfg = {
            k: a
            for k, a in zip(right_hand_robot.actuated_joints, joint_angle_vector)
        }
        hand_positions = right_hand_robot.link_fk(cfg, links = link_names)
        positions = np.array([hand_positions[right_hand_robot.link_map[name]][:3, 3] for name in link_names])
        err = positions - right_hand_tip_poses
        return err.flatten()

    n_joints = len(right_hand_robot.actuated_joints)

    joint_limits_dict = dict(zip(right_hand_robot.actuated_joint_names, right_hand_robot.joint_limits, strict=True))
    lower_bounds = []
    upper_bounds = []
    for joint_name in right_hand_robot.actuated_joint_names:
        lower_bounds.append(joint_limits_dict[joint_name][0])
        upper_bounds.append(joint_limits_dict[joint_name][1])

    jac_sparsity_mat = np.zeros((len(link_names), n_joints), dtype=np.int32)
    # link name index -> joint name index
    jac_sparsity_mat[0, 0] = 1
    jac_sparsity_mat[0, 5] = 1
    jac_sparsity_mat[1, 1] = 1
    jac_sparsity_mat[2, 2] = 1
    jac_sparsity_mat[3, 3] = 1
    jac_sparsity_mat[4, 4] = 1
    # repeat 3 times for each link
    jac_sparsity_mat = np.repeat(jac_sparsity_mat, 3, 0)

    # Display the frame
    optim_res = least_squares(residuals, np.zeros(n_joints), bounds=(lower_bounds, upper_bounds), jac_sparsity=jac_sparsity_mat)
    # print('-'*20)
    # print(f'Average tip position: {np.mean(np.linalg.norm(right_hand_tip_poses, axis=1))}')
    # print("Residuals before:", np.linalg.norm(residuals(np.zeros(n_joints))))
    # print("Residuals after:", np.linalg.norm(residuals(optim_res.x)))
    # print(f"Joint angles: {optim_res.x}")


    return optim_res.x
    # return np.zeros(n_joints)

@app.add_handler("CAMERA_MOVE")
async def on_cam_move(event, session):
    global head_matrix_shared
    head_matrix_shared = np.array(event.value["camera"]["matrix"], dtype=np.float32).reshape(4, 4)


@app.add_handler("HAND_MOVE")
async def hand_move_handler(event, session):
    global left_hand_shared, left_landmarks_shared
    """Handle hand tracking data and print information"""
    if event.key == 'hands':
        if 'leftState' in event.value and event.value['leftState']: # There is also more info in these but we ignore it
            left_mat_raw = event.value['left'] # 400-long float array, 25 4x4 matrices
            left_mat_numpy = np.array(left_mat_raw, dtype=np.float32).reshape(25, 4, 4)
            left_hand_shared[:] = left_mat_numpy[0].T  # Use the first matrix as the hand pose
            left_landmarks_shared[:] = left_mat_numpy[:, 3, :3]

        if 'rightState' in event.value and event.value['rightState']:
            right_mat_raw = event.value['right']
            right_mat_numpy = np.array(right_mat_raw, dtype=np.float32).reshape(25, 4, 4)
            right_hand_shared[:] = right_mat_numpy[0].T  # Use the first matrix as the hand pose
            right_landmarks_shared[:] = right_mat_numpy[:, 3]

    right_tips = right_landmarks_shared#[tip_indices]
    rel_right_tips = right_tips @ fast_mat_inv(right_hand_shared)
    right_tips_urdf_frame = rel_right_tips[:,:3] @ vuer_to_urdf_mat.T

    right_hand_joints = right_hand_inverse_kinematics(right_tips_urdf_frame)

    right_hand_transformed = right_hand_shared.copy()
    right_hand_transformed[:3, :3] = right_hand_transformed[:3, :3] @ vuer_to_urdf_mat


    session.upsert @ Urdf(
        src="https://10.33.12.199/static/inspire_hand/inspire_hand_right.urdf",
        jointValues=dict(zip(right_hand_robot.actuated_joint_names, right_hand_joints, strict=True)),
        matrix = right_hand_transformed.T.flatten().tolist(),
        scale=1,
        key="right_hand",
    )


@app.spawn(start=True)
async def main(sess: VuerSession):
    sess.set @ DefaultScene(
        grid=True,
    )

    sess.upsert(
        Hands(
            stream=True,
            key="hands",
            hideLeft=False,       # hides the hand, but still streams the data.
            hideRight=False,      # hides the hand, but still streams the data.
            # disableLeft=False,    # disables the left data stream, also hides the hand.
            # disableRight=False,   # disables the right data stream, also hides the hand.
        ),
        to="bgChildren",
    )

    await sleep(0.1)

    # Animation variables
    time = 0.0
    dt = 0.016  # 60 FPS

    while True:
        # Simple sinusoidal animations for positions

        left_hand_x = -0.3 + 0.1 * math.sin(time * 0.4)
        left_hand_y = 0.05 * math.cos(time * 0.6)
        left_hand_z = 0.3 + 0.05 * math.sin(time * 0.35)

        kbot_x = 0.1 * math.sin(time * 0.3)
        kbot_y = 0.5 + 0.05 * math.cos(time * 0.5)
        kbot_z = 0.0


        # Animate kbot joint values
        kbot_joint_values = {}
        for i, joint_name in enumerate(kbot_robot.actuated_joint_names):
            frequency = 0.3 + i * 0.05
            amplitude = 0.2 + (i % 4) * 0.1
            phase = i * math.pi / 6
            kbot_joint_values[joint_name] = amplitude * math.sin(time * frequency + phase)

        # Update all three robots

        # sess.upsert @ Urdf(
        #     src="https://10.33.12.199/static/inspire_hand/inspire_hand_left.urdf",
        #     jointValues={k: 0.0 for k in left_hand_robot.actuated_joint_names},
        #     position=[left_hand_x, left_hand_y, left_hand_z],
        #     scale=1,
        #     key="left_hand",
        # )

        # sess.upsert @ Urdf(
        #     src="https://10.33.12.199/static/kbot/robot.urdf",
        #     jointValues=kbot_joint_values,
        #     position=[kbot_x, kbot_y, kbot_z],
        #     scale=1,
        #     key="kbot",
        # )

        await sleep(dt)
        time += dt
