import cv2

import time
from typing import Iterator

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mediapipe as mp
import xml.etree.ElementTree as ET
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from mediapipe import solutions
from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList

p1s1_path = "../dane/Sequences/p1s2/"

mp_drawing = mp.solutions.drawing_utils
pose = mp.solutions.pose.Pose(static_image_mode=True)

NUM_OF_CAMERAS = 4
NUM_OF_LANDMARKS = 33
FPS = 39

Skeleton = list[Landmark]
NormalisedSkeleton = NormalizedLandmarkList

CameraRot = tuple[float,float,float]
CameraPos = tuple[float,float,float]

#                                  camera config
#                                       |
# cam 1 -> video -> stream -> skeleton -+-> normalised skeleton -|
# cam 2 -> video -> stream -> skeleton -+-> normalised skeleton -|
# cam 3 -> video -> stream -> skeleton -+-> normalised skeleton -+-> average skeleton -> gait -> gait sequences -> compressed gait -> identificator
# ...                                   |                        |
# cam n -> video -> stream -> skeleton -+-> normalised skeleton -| 

def plot_skeleton(skeleton: NormalisedSkeleton, pose_connections: list[tuple[int,int]]=None, title: str="", ax: Axes3D=None, show: bool = True, boxsize: float = None):
    if skeleton is None:
        return

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for landmark in skeleton.landmark:
        ax.scatter3D(landmark.x, landmark.y, landmark.z, c="r")

    if pose_connections:
        for a, b in pose_connections:
            ax.plot(
                [ skeleton.landmark[a].x, skeleton.landmark[b].x ], 
                [ skeleton.landmark[a].y, skeleton.landmark[b].y ], 
                [ skeleton.landmark[a].z, skeleton.landmark[b].z ], 
                c="b"
            )

    if boxsize is not None:
        ax.set_xlim(-boxsize, boxsize)
        ax.set_ylim(-boxsize, boxsize)
        ax.set_zlim(-boxsize, boxsize)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    if show:
        plt.show()


def get_camera_calibration(calibration_file_path: str) -> tuple[CameraPos, CameraRot]:
    """
    Retrieves the camera position and orientation from a calibration file.

    The calibration file must be in the format used by the `OpenPose`_ software.

    Parameters:
        calibration_file_path (str): The path to the calibration file.

    Returns:
        tuple: A tuple containing two tuples: camera_pos and camera_rot.
        camera_pos is a tuple of three floats representing the x, y, and z coordinates of the camera in the world coordinate system.
        camera_rot is a tuple of three floats representing the Euler angles (in radians) of the camera.
        The Euler angles represent the rotation of the camera relative to the world coordinate system, with the first angle representing
        the rotation about the x axis, the second about the y axis, and the third about the z axis.
    """

    tree = ET.parse(calibration_file_path)
    root = tree.getroot()
    extrinsic = root.find("Extrinsic")

    tx = extrinsic.attrib['tx']
    ty = extrinsic.attrib['ty']
    tz = extrinsic.attrib['tz']
    rx = extrinsic.attrib['rx']
    ry = extrinsic.attrib['ry']
    rz = extrinsic.attrib['rz']

    return (tx, ty, tz), (rx, ry, rz)


def get_stream_from_video(video_path: str):
    """
    Creates a stream from a video file.

    Parameters:
        video_path (str): The path to the video file.

    Returns:
        Iterator: An iterator over the frames of the video.
    """
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        yield image

    cap.release()


def get_skeleton_from_frame(frame) -> tuple[Skeleton, Skeleton]:
    """
    Extracts the pose landmarks and world landmarks from a given frame.

    Args:
        frame (Matlike): The input frame from which to extract the landmarks.

    Returns:
        tuple[list[Landmark], list[Landmark]]: A tuple containing two lists. The first list contains the pose landmarks extracted from the frame, and the second list contains the corresponding world landmarks.
    """
    results = pose.process(frame)
    return results.pose_landmarks, results.pose_world_landmarks


def get_normalised_skeleton(skeleton: Skeleton, camera_pos: CameraPos, camera_rot: CameraRot) -> NormalisedSkeleton:
    """
    Takes skeleton data, camera position and orientation, and transforms the skeleton data so that it is centered at the origin
    and oriented in the default direction (i.e. the +z axis).

    Parameters:
        skeleton (list): A list containing the pose landmarks extracted from the video
        camera_pos (list): The 3D world coordinates of the camera
        camera_rot (list): The 3D Euler angles (in radians) representing the camera's orientation

    Returns:
        list: A list containing the translated and oriented skeleton data
    """

    t = np.array(camera_pos, dtype=np.float32)
    r = R.from_euler('xyz', camera_rot, degrees=False)

    translated_landmarks = []

    for landmark in skeleton.landmark:
        point = np.array([landmark.x, landmark.y, landmark.z], dtype=np.float32)
        # point = np.subtract(point, t)
        point = r.apply(point, inverse=True)
        translated_landmarks.append(NormalizedLandmark(
            x=point[0], y=point[1], z=point[2], visibility=landmark.visibility
        ))

    return NormalizedLandmarkList(landmark=translated_landmarks)


def get_average_normalised_skeleton(normalised_skeletons) -> NormalisedSkeleton:
    """
    Takes a list of skeleton data, translated and oriented using different cameras, and averages them out to a single skeleton data.

    Parameters:
        skeletons (list): A list of lists containing the pose landmarks extracted from the video, each list being the result of translating and orienting using one camera.

    Returns:
        list: A list containing the averaged out skeleton data
    """

    global NUM_OF_CAMERAS, NUM_OF_LANDMARKS

    x_values = [0] * NUM_OF_LANDMARKS
    y_values = [0] * NUM_OF_LANDMARKS
    z_values = [0] * NUM_OF_LANDMARKS
    v_values = [0] * NUM_OF_LANDMARKS

    num_working_cameras = 0

    for camera_index in range(NUM_OF_CAMERAS):
        skeleton = normalised_skeletons[camera_index]

        if not skeleton:
            continue

        num_working_cameras += 1

        for i, landmark in enumerate(skeleton.landmark):
            x_values[i] += landmark.x
            y_values[i] += landmark.y
            z_values[i] += landmark.z
            v_values[i] += landmark.visibility

    return NormalizedLandmarkList(
        landmark=[
            NormalizedLandmark(
                x=x/num_working_cameras, 
                y=y/num_working_cameras, 
                z=z/num_working_cameras, 
                visibility=v/num_working_cameras
            )
            for x,y,z,v in zip(x_values,y_values,z_values,v_values)
        ]
    )


def single_executor(video_path: str, camera_config_path: str):
    stream = get_stream_from_video(video_path)
    camera_pos, camera_rot = get_camera_calibration(camera_config_path)

    while stream:
        try:
            frame = next(stream)
        except StopIteration:
            print("End of stream")
            break

        skeleton = get_skeleton_from_frame(frame)[1]
        normalised_skeleton = get_normalised_skeleton(skeleton, camera_pos, camera_rot) if skeleton else None

        yield normalised_skeleton
    

def main_executor(videos_path: str, camera_config_path: str):
    video_paths = sorted(list(map(lambda path: os.path.join(videos_path, path), os.listdir(videos_path))))
    camera_config_paths = sorted(list(map(lambda path: os.path.join(camera_config_path, path), os.listdir(camera_config_path))))

    executors = list(map(single_executor, video_paths, camera_config_paths))

    while all(executors):
        try:
            normalised_skeletons = [next(executor) for executor in executors]
        except StopIteration:
            print("End of stream")
            break

        average_normalised_skeleton = get_average_normalised_skeleton(normalised_skeletons)

        yield average_normalised_skeleton


plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

i = 0

for normalised_skeleton in main_executor(
    os.path.join(p1s1_path, "Images"), 
    os.path.join(p1s1_path, "Calibration")
):

    ax.clear()
    plot_skeleton(
        normalised_skeleton, 
        pose_connections=solutions.pose.POSE_CONNECTIONS, 
        title="Average", ax=ax, show=False, boxsize=.5
    )
    plt.savefig(f"./frame_{i}.png")
    plt.pause(1/FPS); i += 1


