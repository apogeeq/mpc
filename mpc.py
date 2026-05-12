import pybullet as p
import pybullet_data
import numpy as np
import cv2
import cv2.aruco as aruco
import scipy.optimize as opt
import matplotlib.pyplot as plt

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

def denormalize(pts_n):

    pts = np.zeros_like(pts_n)

    for i,(x,y) in enumerate(pts_n):

        pts[i,0] = x*fx + cx
        pts[i,1] = y*fy + cy

    return pts

s_des = normalize(s_des_px).flatten()

# ============================================================
# SPECIAL ERROR
# 0 -> 2
# 1 -> 3
# 2 -> 0
# 3 -> 1
# ============================================================

def special_error(current_pts, desired_pts):

    mapping = [2,3,0,1]

    err = 0.0

    for i in range(4):

        j = mapping[i]

        dx = current_pts[i,0] - desired_pts[j,0]
        dy = current_pts[i,1] - desired_pts[j,1]

        err += dx**2 + dy**2

    return np.sqrt(err)

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
        r"triple_pendulum1.urdf",
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

# ============================================================
# TRAJECTORY PLOT
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
        desired_closed[:,0] + 4,
        desired_closed[:,1] + 3,
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

    plt.gca().invert_yaxis()

    plt.xlim([0,width])
    plt.ylim([height,0])

    plt.xlabel("u")
    plt.ylabel("v")

    plt.title("MPC trajectories")

    plt.grid()
    plt.legend()

    plt.show()

# ============================================================
# ERROR PLOT
# ============================================================

def plot_error(errors):

    iterations = np.arange(len(errors))

    plt.figure(figsize=(8,5))

    plt.plot(
        iterations,
        errors,
        linewidth=2
    )

    plt.xlabel("Iteration")
    plt.ylabel("Error")

    plt.title("Tracking error")

    plt.grid()

    plt.show()

# ============================================================
# MPC PARAMETERS
# ============================================================

H = 5
dt = 0.08

lam_u = 0.01
lam_terminal = 15.0

Z = 0.5

# ============================================================
# PREDICTION MODEL
# ============================================================

def predict_features(s0, q0, U):

    s = s0.copy()

    q = q0.copy()

    traj = []

    for k in range(H):

        qdot = U[3*k:3*(k+1)]

        q = q + qdot*dt

        q1,q2,q3 = q

        J = geometric_jacobian(
            q1,
            q2,
            q3
        )

        pts_n = s.reshape(4,2)

        Ls = interaction_matrix(
            pts_n,
            Z
        )

        sdot = Ls @ (J @ qdot)

        s = s + dt*sdot

        traj.append(s.copy())

    return traj

# ============================================================
# COST FUNCTION
# ============================================================

def cost_function(U, s0, q0):

    traj = predict_features(
        s0,
        q0,
        U
    )

    Jcost = 0

    for k,s in enumerate(traj):

        e = s - s_des

        Jcost += e @ e

        qdot = U[3*k:3*(k+1)]

        Jcost += lam_u*(qdot @ qdot)

    e_terminal = traj[-1] - s_des

    Jcost += lam_terminal*(e_terminal @ e_terminal)

    return Jcost

# ============================================================
# IMAGE CONSTRAINTS
# ============================================================

def image_constraints(U, s0, q0):

    traj = predict_features(
        s0,
        q0,
        U
    )

    cons = []

    for s in traj:

        pts = denormalize(
            s.reshape(4,2)
        )

        for (u,v) in pts:

            cons += [
                u - xmin,
                xmax - u,
                v - ymin,
                ymax - v
            ]

    return np.array(cons)

# ============================================================
# MPC
# ============================================================

def run_mpc():

    robot,j1,j2,j3,cam = setup()

    trajectories = []

    errors = []

    initial_pts = None

    while True:

        p.stepSimulation()

        frame = get_camera(robot, cam)

        pts, pts_n = detect_features(frame)

        if pts is not None:

            if initial_pts is None:
                initial_pts = pts.copy()

            trajectories.append(pts.copy())

            # =================================================
            # SPECIAL ERROR
            # =================================================

            err_special = special_error(
                pts,
                s_des_px
            )

            errors.append(err_special)

            s0 = pts_n.flatten()

            q0 = np.array([

                p.getJointState(robot,j1)[0],
                p.getJointState(robot,j2)[0],
                p.getJointState(robot,j3)[0]

            ])

            U0 = np.zeros(3*H)

            bounds = []

            for _ in range(H):

                bounds += [
                    (-2,2),
                    (-2,2),
                    (-2,2)
                ]

            result = opt.minimize(

                fun=lambda U:
                    cost_function(U,s0,q0),

                x0=U0,

                bounds=bounds,

                constraints={
                    'type':'ineq',
                    'fun':lambda U:
                        image_constraints(U,s0,q0)
                },

                method='SLSQP',

                options={
                    'maxiter':50,
                    'ftol':1e-4
                }
            )

            qdot = result.x[:3]

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

            draw(frame, pts, err_special)

            print("error:", err_special)

        cv2.imshow("MPC", frame)

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

    if len(errors) > 0:

        plot_error(errors)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_mpc()
