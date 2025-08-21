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
# urdfpy needs the mesh URLs to be absolute paths so we replace the relative URLS
urdf_path = f'{file_absolute_parent}/assets/inspire_hand/inspire_hand_right.urdf'
urdf_file_contents = open(urdf_path).read()

# Replace both package:/ paths and relative mesh paths with absolute paths
urdf_assets_path = f'{file_absolute_parent}/assets/inspire_hand'
urdf_file_contents = urdf_file_contents.replace('package:/', file_absolute_parent)
urdf_file_contents = urdf_file_contents.replace('./meshes/', f'{urdf_assets_path}/meshes/')

class dumb_hack(io.StringIO):
    name = 'dummy name to get urdfpy to comply'
urdf_file = dumb_hack(urdf_file_contents)

robot = URDF.load(urdf_file)

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
        Movable(
            Urdf(
                src="http://localhost:8012/static/inspire_hand/inspire_hand_right.urdf",
                jointValues={
                    k: 0.0 for k in robot.actuated_joint_names
                },
                key="robot",
            ),
            position=[0, 0, 0.3],
            scale=10,
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

    while True:

        sess.update @ Urdf(
            src="http://localhost:8012/static/inspire_hand/inspire_hand_right.urdf",
            jointValues={
                k: v for k, v in zip(robot.actuated_joint_names, np.zeros(len(robot.actuated_joint_names)))
            },
            key="robot",
        )
        await sleep(0.016)
        i += 1