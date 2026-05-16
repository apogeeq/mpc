import pybullet as p
import pybullet_data
import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import time

# ============================================================
# CAMERA PARAMETERS
# ============================================================

width = 320
height = 320
fov = 60

fx = width / (2 * np.tan(np.deg2rad(fov)/2))
fy = fx

cx = width / 2
cy = height / 2

# ============================================================
# DESIRED IMAGE FEATURES
# ============================================================

size_px = 56

s_des_px = np.array([
    [cx - size_px/2, cy - size_px/2],
    [cx + size_px/2, cy - size_px/2],
    [cx + size_px/2, cy + size_px/2],
    [cx - size_px/2, cy + size_px/2]
])

# ============================================================
# IMAGE CONSTRAINTS
# ============================================================

xmin, xmax = 60, 260
ymin, ymax = 60, 260

# ============================================================
# CORNER COLORS
# ============================================================

corner_colors = [
    (0,0,255),
    (0,255,0),
    (255,0,0),
    (0,255,255)
]

# ============================================================
# ARUCO
# ============================================================

dictionary = aruco.getPredefinedDictionary(
    aruco.DICT_4X4_50
)

detector = aruco.ArucoDetector(dictionary)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(pts):

    pts_n = np.zeros_like(pts, dtype=float)

    for i,(u,v) in enumerate(pts):

        pts_n[i,0] = (u - cx)/fx
        pts_n[i,1] = (v - cy)/fy

    return pts_n

s_des = normalize(s_des_px).flatten()

# ============================================================
# ROBOT GEOMETRIC JACOBIAN
# ============================================================

L1 = 0.4
L2 = 0.4

def geometric_jacobian(q1, q2, q3):

    J = np.zeros((3,3))

    J[0,0] = -L1*np.sin(q1) - L2*np.sin(q1+q2)
    J[0,1] = -L2*np.sin(q1+q2)
    J[0,2] = 0

    J[1,0] = L1*np.cos(q1) + L2*np.cos(q1+q2)
    J[1,1] = L2*np.cos(q1+q2)
    J[1,2] = 0

    J[2,0] = 1
    J[2,1] = 1
    J[2,2] = 1

    return J

# ============================================================
# INTERACTION MATRIX
# ============================================================

def interaction_matrix(pts_n, Z=0.5):

    L = []

    for (x,y) in pts_n:

        Li = np.array([

            [1/Z, 0, y],

            [0, 1/Z, -x]

        ])

        L.append(Li)

    return np.vstack(L)

# ============================================================
# SPECIAL ERROR
# ============================================================

def special_error(pts):

    desired_map = [2,3,0,1]

    err = 0.0

    for i in range(4):

        j = desired_map[i]

        dx = pts[i,0] - s_des_px[j,0]
        dy = pts[i,1] - s_des_px[j,1]

        err += dx**2 + dy**2

    return np.sqrt(err)

# ============================================================
# SETUP
# ============================================================

def setup():

    p.connect(p.GUI)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(0,0,-9.81)

    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf")

    robot = p.loadURDF(
        r"c:\Users\devya\Desktop\nir4\triple_pendulum1.urdf",
        useFixedBase=True
    )

    j1=j2=j3=cam=None

    for i in range(p.getNumJoints(robot)):

        name = p.getJointInfo(robot,i)[1].decode()
        link = p.getJointInfo(robot,i)[12].decode()

        if name=="joint_1":
            j1=i

        if name=="joint_2":
            j2=i

        if name=="joint_3":
            j3=i

        if link=="camera_link":
            cam=i

    # ========================================================
    # MARKER
    # ========================================================

    marker_img = aruco.generateImageMarker(
        dictionary,
        0,
        800
    )

    cv2.imwrite("marker.png", marker_img)

    tex = p.loadTexture("marker.png")

    vis = p.createVisualShape(
        p.GEOM_MESH,
        fileName="marker.obj"
    )

    col = p.createCollisionShape(
        p.GEOM_MESH,
        fileName="marker.obj"
    )

    marker = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0.1,4,2.3]
    )

    p.changeVisualShape(
        marker,
        -1,
        textureUniqueId=tex
    )

    return robot,j1,j2,j3,cam

# ============================================================
# CAMERA
# ============================================================

def get_camera(robot, cam):

    state = p.getLinkState(robot, cam, True)

    pos = state[4]
    orn = state[5]

    R = np.array(
        p.getMatrixFromQuaternion(orn)
    ).reshape(3,3)

    forward = R[:,2]
    up = R[:,1]

    view = p.computeViewMatrix(
        pos,
        pos + forward,
        up
    )

    proj = p.computeProjectionMatrixFOV(
        fov,
        width/height,
        0.01,
        5
    )

    img = p.getCameraImage(
        width,
        height,
        view,
        proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    frame = np.reshape(
        img[2],
        (height,width,4)
    )[:,:,:3]

    frame = np.ascontiguousarray(
        frame.astype(np.uint8)
    )

    return frame

# ============================================================
# DETECTION
# ============================================================

def detect_features(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2GRAY
    )

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:

        return None, None

    pts = corners[0].reshape(4,2)

    pts_n = normalize(pts)

    return pts, pts_n

# ============================================================
# DRAW
# ============================================================

def draw(frame, pts, err):

    cv2.polylines(
        frame,
        [pts.astype(int)],
        True,
        (0,0,255),
        2
    )

    cv2.polylines(
        frame,
        [s_des_px.astype(int)],
        True,
        (0,255,0),
        2
    )

    cv2.rectangle(
        frame,
        (xmin,ymin),
        (xmax,ymax),
        (255,255,0),
        1
    )

    center = np.mean(pts, axis=0)

    cv2.circle(
        frame,
        center.astype(int),
        4,
        (255,0,0),
        -1
    )

    cv2.putText(
        frame,
        f"err={err:.2f}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,0,0),
        2
    )

# ============================================================
# PLOT TRAJECTORIES
# ============================================================

def plot_trajectories(initial_pts, traj):

    plt.figure(figsize=(8,8))

    constraint = np.array([
        [xmin,ymin],
        [xmax,ymin],
        [xmax,ymax],
        [xmin,ymax],
        [xmin,ymin]
    ])

    plt.plot(
        constraint[:,0],
        constraint[:,1],
        'c-',
        linewidth=2,
        label='Constraint'
    )

    initial_closed = np.vstack([
        initial_pts,
        initial_pts[0]
    ])

    plt.plot(
        initial_closed[:,0],
        initial_closed[:,1],
        'r-',
        linewidth=2,
        label='Initial marker'
    )

    desired_closed = np.vstack([
        s_des_px,
        s_des_px[0]
    ])

    plt.plot(
        desired_closed[:,0]+4,
        desired_closed[:,1]+2,
        'g-',
        linewidth=2,
        label='Desired marker'
    )

    traj = np.array(traj)

    for i in range(4):

        plt.plot(
            traj[:,i,0],
            traj[:,i,1],
            linewidth=2,
            label=f'Corner {i}'
        )

        plt.scatter(
            traj[0,i,0],
            traj[0,i,1],
            s=50
        )

    plt.gca().invert_yaxis()

    plt.xlim([0,width])
    plt.ylim([height,0])

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Image Feature Trajectories")

    plt.legend()

    plt.grid()

    plt.show()

# ============================================================
# ERROR PLOT
# ============================================================

def plot_error(times, errors):

    plt.figure(figsize=(8,4))

    plt.plot(
        times,
        errors,
        linewidth=2
    )

    plt.xlabel("Time [s]")

    plt.ylabel("Error")

    plt.title("Error convergence")

    plt.grid()

    plt.show()

# ============================================================
# IBVS
# ============================================================

def run_ibvs():

    robot,j1,j2,j3,cam = setup()

    lam = 4.0
    mu = 0.01
    Z = 0.5

    trajectories = []

    initial_pts = None

    error_history = []
    time_history = []

    start_time = time.time()

    while True:

        p.stepSimulation()

        frame = get_camera(robot, cam)

        pts, pts_n = detect_features(frame)

        if pts is not None:

            if initial_pts is None:
                initial_pts = pts.copy()

            trajectories.append(pts.copy())

            s = pts_n.flatten()

            e = s - s_des

            special_err = special_error(pts)

            error_history.append(special_err)

            current_time = time.time() - start_time

            time_history.append(current_time)

            Ls = interaction_matrix(
                pts_n,
                Z
            )

            Ls_pinv = (
                Ls.T @
                np.linalg.inv(
                    Ls @ Ls.T +
                    mu*np.eye(8)
                )
            )

            vc = -lam * (Ls_pinv @ e)

            q1 = p.getJointState(robot,j1)[0]
            q2 = p.getJointState(robot,j2)[0]
            q3 = p.getJointState(robot,j3)[0]

            J = geometric_jacobian(
                q1,
                q2,
                q3
            )

            J_pinv = (
                J.T @
                np.linalg.inv(
                    J @ J.T +
                    mu*np.eye(3)
                )
            )

            qdot = J_pinv @ vc

            qdot = np.clip(
                qdot,
                -2.0,
                2.0
            )

            p.setJointMotorControl2(
                robot,
                j1,
                p.VELOCITY_CONTROL,
                targetVelocity=qdot[0],
                force=200
            )

            p.setJointMotorControl2(
                robot,
                j2,
                p.VELOCITY_CONTROL,
                targetVelocity=qdot[1],
                force=200
            )

            p.setJointMotorControl2(
                robot,
                j3,
                p.VELOCITY_CONTROL,
                targetVelocity=qdot[2],
                force=100
            )

            draw(frame, pts, special_err)

            print("error:", special_err)

        cv2.imshow("IBVS", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):

            break

    p.disconnect()

    cv2.destroyAllWindows()

    if initial_pts is not None:

        plot_trajectories(
            initial_pts,
            trajectories
        )

        plot_error(
            time_history,
            error_history
        )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_ibvs()
