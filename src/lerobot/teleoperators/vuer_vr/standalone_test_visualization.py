import math
from asyncio import sleep
from pathlib import Path

from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Urdf, Movable, Hands


from urdfpy import URDF
import io
import numpy as np

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
left_hand_robot = load_urdf_with_absolute_paths('assets/inspire_hand/inspire_hand_left.urdf', 'assets/inspire_hand')
kbot_robot = load_urdf_with_absolute_paths('assets/kbot/robot.urdf', 'assets/kbot')

app = Vuer(static_root=Path(__file__).parent / "assets")

head_matrix_shared = np.zeros((4, 4), dtype=np.float32)
left_hand_shared = np.zeros((4, 4), dtype=np.float32)
left_landmarks_shared = np.zeros((25 * 3,), dtype=np.float32)

right_hand_shared = np.zeros((4, 4), dtype=np.float32)
right_landmarks_shared = np.zeros((25 * 3,), dtype=np.float32)

@app.add_handler("CAMERA_MOVE")
async def on_cam_move(event, session):
    global head_matrix_shared
    head_matrix_shared = np.array(event.value["camera"]["matrix"], dtype=np.float32).reshape(4, 4)

@app.add_handler("HAND_MOVE")
async def hand_move_handler(event, session):
    global left_hand_shared, left_landmarks_shared
    """Handle hand tracking data and print information"""
    if event.key == 'hands':
        if event.value['leftState']: # There is also more info in these but we ignore it
            left_mat_raw = event.value['left'] # 400-long float array, 25 4x4 matrices
            left_mat_numpy = np.array(left_mat_raw, dtype=np.float32).reshape(25, 4, 4)
            left_hand_shared[:] = left_mat_numpy[0].T  # Use the first matrix as the hand pose
            left_landmarks_shared[:] = left_mat_numpy[:, 3, :3].flatten()

        if event.value['rightState']:
            right_mat_raw = event.value['right']
            right_mat_numpy = np.array(right_mat_raw, dtype=np.float32).reshape(25, 4, 4)
            right_hand_shared[:] = right_mat_numpy[0].T  # Use the first matrix as the hand pose
            right_landmarks_shared[:] = right_mat_numpy[:, 3, :3].flatten()


@app.spawn(start=True)
async def main(sess: VuerSession):
    sess.set @ DefaultScene(
        # Right hand
        Movable(
            Urdf(
                src="http://localhost:8012/static/inspire_hand/inspire_hand_right.urdf",
                jointValues={
                    k: 0.0 for k in right_hand_robot.actuated_joint_names
                },
                key="right_hand",
            ),
            position=[0.3, 0, 0.3],
            scale=10,
        ),
        # Left hand
        Movable(
            Urdf(
                src="http://localhost:8012/static/inspire_hand/inspire_hand_left.urdf",
                jointValues={
                    k: 0.0 for k in left_hand_robot.actuated_joint_names
                },
                key="left_hand",
            ),
            position=[-0.3, 0, 0.3],
            scale=10,
        ),
        # Kbot robot
        Movable(
            Urdf(
                src="http://localhost:8012/static/kbot/robot.urdf",
                jointValues={
                    k: 0.0 for k in kbot_robot.actuated_joint_names
                },
                key="kbot",
            ),
            position=[0, 0.5, 0.0],
            scale=1,
        ),
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
        right_hand_x = 0.3 + 0.1 * math.sin(time * 0.5)
        right_hand_y = 0.05 * math.cos(time * 0.7)
        right_hand_z = 0.3 + 0.05 * math.sin(time * 0.3)

        left_hand_x = -0.3 + 0.1 * math.sin(time * 0.4)
        left_hand_y = 0.05 * math.cos(time * 0.6)
        left_hand_z = 0.3 + 0.05 * math.sin(time * 0.35)

        kbot_x = 0.1 * math.sin(time * 0.3)
        kbot_y = 0.5 + 0.05 * math.cos(time * 0.5)
        kbot_z = 0.0

        # Animate right hand joint values with appropriate ranges
        right_hand_joint_values = {}
        for i, joint in enumerate(right_hand_robot.actuated_joints):
            frequency = 0.5 + i * 0.1
            phase = i * math.pi / 4
            
            # Use the actual joint limits for realistic animation
            joint_range = joint.limit.upper - joint.limit.lower
            joint_center = (joint.limit.upper + joint.limit.lower) / 2
            amplitude = joint_range * 0.4  # Use 40% of the joint range
            
            value = joint_center + amplitude * math.sin(time * frequency + phase)
            right_hand_joint_values[joint.name] = max(joint.limit.lower, min(joint.limit.upper, value))

        # Animate left hand joint values with appropriate ranges
        left_hand_joint_values = {}
        for i, joint in enumerate(left_hand_robot.actuated_joints):
            frequency = 0.6 + i * 0.1
            phase = i * math.pi / 4 + math.pi / 2  # Phase offset for variety

            # Use the actual joint limits for realistic animation
            joint_range = joint.limit.upper - joint.limit.lower
            joint_center = (joint.limit.upper + joint.limit.lower) / 2
            amplitude = joint_range * 0.4  # Use 40% of the joint range

            value = joint_center + amplitude * math.sin(time * frequency + phase)
            left_hand_joint_values[joint.name] = max(joint.limit.lower, min(joint.limit.upper, value))

        # Make the first finger very obvious for debugging
        if len(left_hand_joint_values) > 1:
            first_joint_name = list(left_hand_joint_values.keys())[1]  # Use index finger
            left_hand_joint_values[first_joint_name] = 0.85 * (1 + math.sin(time * 2))  # Very obvious animation

        # Animate kbot joint values
        kbot_joint_values = {}
        for i, joint_name in enumerate(kbot_robot.actuated_joint_names):
            frequency = 0.3 + i * 0.05
            amplitude = 0.2 + (i % 4) * 0.1
            phase = i * math.pi / 6
            kbot_joint_values[joint_name] = amplitude * math.sin(time * frequency + phase)

        # Update all three robots
        sess.upsert @ Movable(
            Urdf(
                src="http://localhost:8012/static/inspire_hand/inspire_hand_right.urdf",
                jointValues=right_hand_joint_values,
                key="right_hand",
            ),
            position=[right_hand_x, right_hand_y, right_hand_z],
            scale=10,
        )

        sess.upsert @ Movable(
            Urdf(
                src="http://localhost:8012/static/inspire_hand/inspire_hand_left.urdf",
                jointValues=left_hand_joint_values,
                key="left_hand",
            ),
            position=[left_hand_x, left_hand_y, left_hand_z],
            scale=10,
        )

        sess.upsert @ Movable(
            Urdf(
                src="http://localhost:8012/static/kbot/robot.urdf",
                jointValues=kbot_joint_values,
                key="kbot",
            ),
            position=[kbot_x, kbot_y, kbot_z],
            scale=1,
        )

        await sleep(dt)
        time += dt
