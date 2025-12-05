import os
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation


def traj_linear_interpolation(x_start: np.ndarray, x_end: np.ndarray, N: int):
    assert x_start.shape == x_end.shape, "Start and end states must have the same dimensions."
    traj = np.zeros((N, x_start.shape[0]))
    for i in range(N):
        alpha = i / (N - 1)
        traj[i] = (1 - alpha) * x_start + alpha * x_end
    return traj

def create_desired_trajectory(N, dt, padding_N=None):
    
    N_pre = int(0.1 * N)
    N_post = int(0.25 * N)
    N_circle = N - N_pre - N_post 
    
    # in real application, x0 should be the current state of the chaser spacecraft
    # but here we just define it statically for simplicity
    x0 = np.array([-2, -4, 2, 0, 0, 0]) # initial state
    xg = np.array([0, -0.68, 3.05, 0, 0, 0])   # goal state
    
    # create a circular trajectory in 3D space
    circle_center = np.array([0.0, -3.0, 3.0])
    radius = 1.1
    theta_vec = np.linspace(-np.pi/2, 2*np.pi + np.pi/2, N_circle+2)
    pos_circle = [
        [radius*np.cos(theta) + circle_center[0],
         circle_center[1],
         radius*np.sin(theta) + circle_center[2]] for theta in theta_vec
    ]
    pos_circle = np.array(pos_circle)
    
    pos_pre_circle = traj_linear_interpolation(x0[:3], pos_circle[0], N_pre)
    
    N_post1 = N_post // 3
    N_post2 = N_post - N_post1 
    
    start_post = pos_circle[-1]
    mid_post = np.array([xg[0], -3, xg[2]])
    end_post = np.array(xg[:3])
    
    pos_post1 = traj_linear_interpolation(start_post, mid_post, N_post1)
    pos_post2 = traj_linear_interpolation(mid_post, end_post, N_post2)
    pos_post_circle = np.vstack((pos_post1, pos_post2))
    
    pos = np.vstack((pos_pre_circle, pos_circle[1:-1], pos_post_circle))
    
    if padding_N is not None:
        pad_end = np.tile(pos[-1], (padding_N, 1))
        pos = np.vstack((pos, pad_end))
    
    vel = np.zeros_like(pos)
    for i in range(1, len(pos)):
        vel[i] = (pos[i] - pos[i-1]) / dt

    traj = np.hstack((pos, vel))
    assert traj.shape == (N + (padding_N if padding_N is not None else 0), 6), "Trajectory shape mismatch."
    
    return traj
    
    
def vis_traj(vis, name, X, color=0xff0000, linewidth=2):
    # X shape: (N, 6) or (N, 3)
    points = X[:, :3].T.astype(np.float32)
    
    material = g.LineBasicMaterial(color=color, linewidth=linewidth)
    geometry = g.PointsGeometry(points)

    vis[name].set_object(g.Line(geometry, material))
    

def animate_rendezvous(X, X_ref, dt, show_reference=True):

    vis = meshcat.Visualizer()
    
    vis["/Background"].set_property("top_color", [0, 0, 0])
    vis["/Background"].set_property("bottom_color", [0.1, 0.1, 0.2])
    
    scale_dragon = 0.002
    rot_dragon_base = tf.rotation_matrix(np.pi/2, [1, 0, 0]) 
    trans_dragon_base = tf.translation_matrix([0, 0, -0.34])
    T_dragon = trans_dragon_base @ rot_dragon_base @ tf.scale_matrix(scale_dragon)

    try:
        dragon_geo = g.ObjMeshGeometry.from_file("models/dragon.obj")
        vis["dragon"]["base"].set_object(dragon_geo, g.MeshPhongMaterial(color=0x9999ff))
    except:
        print("Warning: dragon.obj not found, using Box placeholder.")
        vis["dragon"]["base"].set_object(g.Box([0.5, 0.5, 0.5]), g.MeshPhongMaterial(color=0x9999ff))
    
    vis["dragon"]["base"].set_transform(T_dragon)

    scale_iss = 0.023
    rot_iss_base = tf.rotation_matrix(np.pi/2, [1, 0, 0])
    trans_iss_base = tf.translation_matrix([-8.915, -2.5, 4])
    T_iss = trans_iss_base @ rot_iss_base @ tf.scale_matrix(scale_iss)

    try:
        iss_geo = g.ObjMeshGeometry.from_file("models/ISS.obj")
        vis["iss"]["base"].set_object(iss_geo, g.MeshPhongMaterial(color=0x999999))
    except:
        print("Warning: ISS.obj not found, using Box placeholder.")
        vis["iss"]["base"].set_object(g.Box([2, 2, 2]), g.MeshPhongMaterial(color=0x999999))
        
    vis["iss"]["base"].set_transform(T_iss)

    if show_reference:
        vis_traj(vis, "traj", X_ref, color=0xff0000)

    anim = Animation(default_framerate=1.0/dt)
    
    for k in range(len(X)):
        pos = X[k, :3]

        rot_z = tf.rotation_matrix(np.pi, [0, 0, 1])
        trans = tf.translation_matrix(pos)
        
        T_anim = trans @ rot_z
        
        with anim.at_frame(vis, k) as frame:
            frame["dragon"].set_transform(T_anim)

    vis.set_animation(anim)
    
    vis.open()
    
    return vis
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 100
    dt = 1.0
    padding_N = 20
    traj = create_desired_trajectory(N, dt, padding_N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:,0], traj[:,1], traj[:,2], label='Desired Trajectory')
    ax.scatter(traj[0,0], traj[0,1], traj[0,2], color='green', label='Start')
    ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='red', label='Goal')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.legend()
    plt.show()