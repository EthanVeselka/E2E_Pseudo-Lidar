import sys
import carla
import automatic_control as ac # Taken from Carla's example code
import generate_traffic as gt # Taken from Carla's example code
import argparse
import logging
import pygame
import random
import time
import datetime
import os
import numpy as np
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
import configparser
import threading
import shutil
from math import tan, pi, acos
from kitti_utils import write_episode_kitti
import copy

from agents.navigation.behavior_agent import BehaviorAgent 
from agents.navigation.basic_agent import BasicAgent  
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  

conf = configparser.ConfigParser()
conf.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))

CARLA_PYTHON_PATH = conf["Paths"]["CARLA_PYTHON_PATH"]
if CARLA_PYTHON_PATH not in sys.path:
    sys.path.insert(0,CARLA_PYTHON_PATH)
    
DATA_PATH = conf["Paths"]["DATA_PATH"]

POLL_RATE = float(conf["Settings"]["POLL_RATE"])
CAMERA_X = int(conf["Settings"]["CAMERA_X"])
CAMERA_Y = int(conf["Settings"]["CAMERA_Y"])
CAMERA_FOV = int(conf["Settings"]["CAMERA_FOV"])

EGO_BEHAVIOR = conf["Internal Variables"]["EGO_BEHAVIOR"]
EXTERNAL_BEHAVIOR = conf["External Variables"]["EXTERNAL_BEHAVIOR"]
WEATHER = int(conf["External Variables"]["WEATHER"])
MAP = conf["External Variables"]["MAP"]

FILES_PER_FRAME = 6
DEBUG_ON = False

output_path = ''

position_dict = {}
actor_bb_dict = {}
first_data_frame = None
actors = None

l_rgb_dict = {}
r_rgb_dict = {}
d_rgb_dict = {}
s_lidar_dict = {}

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

def get_camera_point(loc, w2c):
        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        return point_camera

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def build_calibration_mat(values):
    flip = np.array([[ 0, 1, 0 ], [ 0, 0, -1 ], [ 1, 0, 0 ]], dtype=np.float32)

    x = values['x']
    y = values['y']
    z = values['z']
    pitch = values['pitch']
    roll = values['roll']
    yaw = values['yaw']
    fov = CAMERA_FOV
    Cx = CAMERA_X / 2
    Cy = CAMERA_Y / 2

    f = CAMERA_X /(2.0 * tan(fov * pi / 360))
    K = np.array([[f, 0, Cx], [0, f, Cy], [0, 0, 1]], dtype=np.float64)
    

    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    matrix = np.linalg.inv(matrix)
    
    P = K @ flip @ matrix[:3, :]
    return P

def save_calibration_mats(p0, p1):

    def write_mat(mat, dest):
        for row in mat:
            for elem in row:
                dest.write(str(elem) + ' ')
            dest.write('\n')


    r0 = np.identity(3)
    
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
    TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))
    with open(os.path.join(DATA_PATH, "calibmatrices.txt"), 'w') as dest:
        dest.write('P0:\n')
        write_mat(p0, dest)
        dest.write('P1:\n')
        write_mat(p1, dest)
        dest.write('R0_rect:\n')
        write_mat(r0, dest)
        dest.write('tr_velodyne_to_cam:\n')
        write_mat(TR_velodyne, dest)

def save_box(bb, bbox_elem, sensor_transform, world_2_camera, object_transform = None):

                
                edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

                K = build_projection_matrix(CAMERA_X, CAMERA_Y, CAMERA_FOV)

                is_dynamic = True
                if object_transform is None:
                    object_transform = carla.Transform()
                    is_dynamic = False

                verts = [v for v in bb.get_world_vertices(object_transform)]

                x_max = -10000
                x_min = 10000
                y_max = -10000
                y_min = 10000
                for vert in verts:
                    p = get_image_point(vert, K, world_2_camera)
                    if p[0] > x_max:
                        x_max = p[0]
                    if p[0] < x_min:
                        x_min = p[0]
                    if p[1] > y_max:
                        y_max = p[1]
                    if p[1] < y_min:
                        y_min = p[1]

                # Add the object to the frame (ensure it is inside the image)
                on_screen = ET.SubElement(bbox_elem, "on_screen")
                if not (x_min > 0 and x_max < CAMERA_X and y_min > 0 and y_max < CAMERA_Y):
                    on_screen.set('on_screen', 'true')
                else:
                    on_screen.set('on_screen', 'false')

                # Store center of object
                bbox_elem_center = ET.SubElement(bbox_elem, "center")
                bbox_elem_center.set("x", str(bb.location.x))
                bbox_elem_center.set("y", str(bb.location.y))
                bbox_elem_center.set("z", str(bb.location.z))

                # Store bb verticies in camera-space
                bbox_elem_verts = ET.SubElement(bbox_elem, "verts")
                camera_verts = [v for v in bb.get_world_vertices(sensor_transform)]
                for i in range(len(verts)):
                    vert = camera_verts[i]
                    elem_vert = ET.SubElement(bbox_elem_verts, "v" + str(i))
                    elem_vert.set("x", str(vert.x))
                    elem_vert.set("y", str(vert.y))
                    elem_vert.set("z", str(vert.z))
                
                # Store shape of bb
                bbox_elem_shape = ET.SubElement(bbox_elem, "shape")
                bbox_elem_shape.set('x', str(bb.extent.x * 2))
                bbox_elem_shape.set('y', str(bb.extent.y * 2))
                bbox_elem_shape.set('z', str(bb.extent.z * 2))

                # Store world orientation of bb
                bbox_elem_rot = ET.SubElement(bbox_elem, "word_orientation")
                bbox_elem_rot.set('pitch', str(object_transform.rotation.pitch))
                bbox_elem_rot.set('yaw', str(object_transform.rotation.yaw))
                bbox_elem_rot.set('roll', str(object_transform.rotation.roll))

                # Store orientation of bb relative to camera
                bbox_elem_rel = ET.SubElement(bbox_elem, "relative_orientation")
                bbox_elem_rel.set('pitch', str(object_transform.rotation.pitch - sensor_transform.rotation.pitch))
                bbox_elem_rel.set('yaw', str(object_transform.rotation.yaw - sensor_transform.rotation.pitch))
                bbox_elem_rel.set('roll', str(object_transform.rotation.roll - sensor_transform.rotation.pitch))

                # Store location of bb center relative to sensor
                obj_location = bb.location
                if is_dynamic:
                    obj_location = object_transform.location
                centroid_location = get_camera_point(obj_location, world_2_camera)
                bbox_elem_rel_loc = ET.SubElement(bbox_elem, "relative_center")
                bbox_elem_rel_loc.set('x', str(centroid_location[0]))
                bbox_elem_rel_loc.set('y', str(centroid_location[1]))
                bbox_elem_rel_loc.set('z', str(centroid_location[2]))
                ray_vector = np.array([centroid_location[0], centroid_location[2]])

                # Store observation angle of object center to camera-x_axis
                forward_vec = bb.rotation.get_forward_vector()
                if is_dynamic:
                    forward_vec = object_transform.get_forward_vector()
                obj_vector3D = sensor_transform.transform_vector(forward_vec)
                obj_vector = np.array([obj_vector3D.x, obj_vector3D.z])
                angle = acos(np.dot(ray_vector, obj_vector) / np.linalg.norm(ray_vector) * np.linalg.norm(obj_vector))
                bbox_elem_center = bbox_elem_rel.set('observation_angle', str(angle))
                
                # Store 2D Box values
                bbox_elem_2d_box = ET.SubElement(bbox_elem, "Box2d")
                bbox_elem_2d_box.set("xMin", str(x_min))
                bbox_elem_2d_box.set("xMax", str(x_max))
                bbox_elem_2d_box.set("yMin", str(y_min))
                bbox_elem_2d_box.set("yMax", str(y_max))
                
                counter = 0
                for edge in edges:
                    # Join the vertices into edges
                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
                    
                    bbox_elem_edge = ET.SubElement(bbox_elem, "edge" + str(counter))
                    
                    bbox_elem_edge.set("x1", str(p1[0]))
                    bbox_elem_edge.set("y1", str(p1[1]))
                    bbox_elem_edge.set("x2", str(p2[0]))
                    bbox_elem_edge.set("y2", str(p2[1]))
                    
                    counter += 1

            
def save_boxes(world, sample_path, transform, frame_num):
    global position_dict
    global first_data_frame
    
    #print(world.player.get_transform().location)
    # Create the XML structure
    
    root = ET.Element("StaticBoundingBoxes")
    tree = ET.ElementTree(root)
    #bbs = world.world.get_level_bbs(carla.CityObjectLabel.Car)
    #print('looking for:', frame_num)
    
    
    camera_transform = transform
    # get accurate camera position from position_dict
    for i in range(2):
        try:
            saved_frame_num = frame_num - first_data_frame+1
            if first_data_frame is not None and saved_frame_num in position_dict:
                camera_transform = position_dict[saved_frame_num]
                break
            else:
                time.sleep(1)
        except:
            time.sleep(1)
            
    #print(bounding_box_set)
    
    filters = [
        carla.CityObjectLabel.Buildings,
        carla.CityObjectLabel.Fences,
        carla.CityObjectLabel.Poles,
        carla.CityObjectLabel.RoadLines,
        carla.CityObjectLabel.Roads,
        carla.CityObjectLabel.Sidewalks,
        carla.CityObjectLabel.Vegetation,
        carla.CityObjectLabel.Walls,
        carla.CityObjectLabel.Sky,
        carla.CityObjectLabel.Ground,
        carla.CityObjectLabel.Bridge,
        carla.CityObjectLabel.RailTrack,
        carla.CityObjectLabel.GuardRail,
        carla.CityObjectLabel.Water,
        carla.CityObjectLabel.Terrain,
        carla.CityObjectLabel.TrafficLight,
        carla.CityObjectLabel.TrafficSigns,
        ]
        
    
    world_2_camera = np.array(camera_transform.get_inverse_matrix())
    bounding_box_set = []
    
    for obj in filters:
        new_bbs = world.get_level_bbs(obj)
        for bb in new_bbs:
            bounding_box_set.append((obj, bb))

    for label, bb in bounding_box_set:
        
        # Filter for distance from ego vehicle
        if bb.location.distance(camera_transform.location) < 50:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the bounding box. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = camera_transform.get_forward_vector()
            ray = bb.location - camera_transform.location
            
            if forward_vec.dot(ray) > 1:
                # save the box
                bbox_elem = ET.SubElement(root, "BoundingBox")
                bbox_elem.set("class", str(label))
                save_box(bb, bbox_elem, camera_transform, world_2_camera)

    # Save the bounding boxes in the scene
    filename = 'static_bbs.xml'
    file_path = os.path.join(sample_path, filename)
    indent(root)
    tree.write(file_path)

                

def rgb_callback(data, name, episode_path, world, player_transform):
    global first_data_frame
    sample_path = os.path.join(episode_path, str(data.frame))
    
    file_name = '%s.png' % name
    full_path = os.path.join(sample_path, file_name)

    # data.save_to_disk(full_path)
    
    if first_data_frame is None:
        first_data_frame = data.frame
        
    if name == 'left_rgb':
        l_rgb_dict[data.frame] = (data, sample_path, full_path)
    if name == 'right_rgb':
        r_rgb_dict[data.frame] = (data, sample_path, full_path)
    
def depth_callback(data, name, episode_path):
    sample_path = os.path.join(episode_path, str(data.frame))

    file_name = '%s.png' % name
    full_path = os.path.join(sample_path, file_name)
    
    d_rgb_dict[data.frame] = (data, sample_path, full_path)
    
    # data.save_to_disk(full_path, color_converter=carla.ColorConverter.Depth)

def lidar_callback(data, name, episode_path, actors, bb_transform_dict):
    sample_path = os.path.join(episode_path, str(data.frame))

    file_name = '%s.ply' % name
    full_path = os.path.join(sample_path, file_name)
    
    s_lidar_dict[data.frame] = (data, sample_path, full_path)    

def start_process(name, data_dict, process_function, world):
    sub_process_threads = []
    for key, value in data_dict.items():
        sub_process_threads.append(threading.Thread(target=process_function, args=(name, value[0], value[1], value[2], world)))
        
    for thread in sub_process_threads:
        thread.start()
    
    for thread in sub_process_threads:
        thread.join()

def rgb_process(name, data, sample_path, full_path, world):
    try:
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)
    except:
        None

    data.save_to_disk(full_path)
    
    if name == 'left_rgb':
        save_boxes(world, sample_path, data.transform, data.frame)

def depth_process(name, data, sample_path, full_path, world):
    try:
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)
    except:
        None
        
    data.save_to_disk(full_path, color_converter=carla.ColorConverter.Depth)

def lidar_process(name, data, sample_path, full_path, world):
    try:
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)
    except:
        None
        
    with open(full_path, 'w') as file:
                    file.write("ply\n"
                    "format ascii 1.0\n"
                    "element vertex " + str(len(data)) + "\n"
                    "property float32 x\n"
                    "property float32 y\n"
                    "property float32 z\n"
                    "property float32 CosAngle\n"
                    "property uint32 ObjIdx\n"
                    "property uint32 ObjTag\n"
                    "end_header\n")
                    
                    for point in data:
                        file.write(str(point.point.x) + " "  + 
                                    str(point.point.y * -1.0) + " " + 
                                    str(point.point.z) + " " +
                                    str(point.cos_inc_angle) + " " +
                                    str(point.object_idx) + " " +
                                    str(point.object_tag) + "\n")
    actor_set = set()
    
    try:
        bb_transform_dict = actor_bb_dict[data.frame]
    except:
        print("Frame", data.frame, "missing from actor_bb_dict")
        return
        
    # Get unique actors from lidar data
    for point in data:
        if point.object_idx:
            actor_set.add(point.object_idx)
    
    # All unique edges needed to create bounding box
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    camera_transform = data.transform
    # get accurate camera position from position_dict
    for i in range(2):
        try:
            saved_frame_num = data.frame - first_data_frame+1
            if first_data_frame is not None and saved_frame_num in position_dict:
                camera_transform = position_dict[saved_frame_num]
                break
            else:
                time.sleep(1)
        except:
            time.sleep(1)
    
    world_2_camera = np.array(camera_transform.get_inverse_matrix())
    K = build_projection_matrix(CAMERA_X, CAMERA_Y, CAMERA_FOV)
    
    root = ET.Element("DynamicBoundingBoxes")
    tree = ET.ElementTree(root)

    for actor_id in actor_set:
        actor = actors.find(actor_id)
        
        # Filter ego vehicle out of bounding box output
        if (actor.attributes.get('role_name') == 'hero'):
            continue
        
        bbox_elem = ET.SubElement(root, "BoundingBox")
        # bbox_elem.set("class", str(type(actor)).split("'")[1])
        
        if "base_type" in actor.attributes and actor.attributes["base_type"] != "":
            bbox_elem.set("class", actor.attributes["base_type"].capitalize())
        else:
            bbox_elem.set("class", str(type(actor)).split("'")[1].split(".")[-1])
            
        position_transform = bb_transform_dict[actor_id][1]

        good_frame = False
        world_2_camera = np.array(data.transform.get_inverse_matrix())
        
        for i in range(2):
            if first_data_frame is None:
                print('no first frame')
                time.sleep(1)
                continue
            saved_frame_num = data.frame-first_data_frame + 1
            if saved_frame_num not in position_dict:
                print('not in dict')
                time.sleep(1)
                continue
            good_frame = True
            world_2_camera = np.array(position_dict[saved_frame_num].get_inverse_matrix())

        if not good_frame:
            #print('lidar frame oofed:', data.frame)
            pass
        
        
        bbox_elem.set("actorId", str(actor_id))

        bb = bb_transform_dict[actor_id][0]
        object_transform = bb_transform_dict[actor_id][1]

        save_box(bb, bbox_elem, camera_transform, world_2_camera, object_transform)
    
    # Save the bounding boxes in the scene
    filename = 'dynamic_bbs.xml'
    file_path = os.path.join(sample_path, filename)
    indent(root)
    tree.write(file_path)

def prep_episode(client, args, iteration_name, episode_name): # uses code from automatic_control.py and generate_traffic.py
    """
    Does setup for a simulation episode, including: setup cameras, setup pygame
    window, setup player and player behavior
    """
    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)


        traffic_manager = client.get_trafficmanager(args.tm_port)
        sim_world = client.get_world()
        settings = sim_world.get_settings()
        #settings.fixed_delta_seconds = 1/POLL_RATE
        
        if not args.asynch:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = ac.HUD(args.width, args.height)
        
        
        world = ac.World(client.get_world(), hud, args)
        controller = ac.KeyboardControl(world)
        
        
        
        if args.agent == "Basic":
            agent = BasicAgent(world.player, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player, 30)
            ground_loc = world.world.ground_projection(world.player.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player, behavior=args.behavior) 

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)
        
    
        
        
        #sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        # Create data paths and directories
        global output_path
        
        ts = datetime.datetime.now()
        data_path = os.path.join(DATA_PATH) 
        episode_path = os.path.join(data_path, episode_name)
        iteration_path = os.path.join(episode_path, iteration_name)
        output_path = os.path.join(iteration_path, str(ts).replace(':', '-').replace('.', '-').replace(' ', '_'))
        
        if not os.path.exists(episode_path):
            os.mkdir(episode_path)
        
        if not os.path.exists(iteration_path):
            os.mkdir(iteration_path)
        
        os.mkdir(output_path)
        
        # Store config in data directories
        with open(os.path.join(episode_path, "config.ini"), 'w') as dest:
            config = "[Internal Variables]\nEGO_BEHAVIOR = " + EGO_BEHAVIOR
            dest.write(config)
        
        with open(os.path.join(output_path, "config.ini"), 'w') as dest:
            config = "[External Variables]\nEXTERNAL_BEHAVIOR = " + EXTERNAL_BEHAVIOR + "\nWEATHER = " + str(WEATHER) + "\nMAP = " + MAP
            dest.write(config)

        # get camera calibration matricies
        left_rgb_vals = {
                'x' : 0,
                'y' : 0,
                'z' : 0,
                'pitch' : 0,
                'roll' : 0,
                'yaw' : 0,
        }
        left_mat = build_calibration_mat(left_rgb_vals)

        right_rgb_vals = {
                'x' : 0,
                'y' : 0.5,
                'z' : 0,
                'pitch' : 0,
                'roll' : 0,
                'yaw' : 0,
        }
        right_mat = build_calibration_mat(right_rgb_vals)

        # save camera calibration to data directory
        save_calibration_mats(left_mat, right_mat)
                
       
        bp_library = world.world.get_blueprint_library()
        rgb_bp = bp_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(CAMERA_X))
        rgb_bp.set_attribute('image_size_y', str(CAMERA_Y))
        rgb_bp.set_attribute('sensor_tick', str(1/POLL_RATE))
        
        transform = carla.Transform(carla.Location(x=0.60, y=-0.25, z=1.8))
        l_rgb = world.world.spawn_actor(rgb_bp, transform, attach_to=world.player, attachment_type=carla.AttachmentType.Rigid)
        
        
        
        
        rgb_bp = bp_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(CAMERA_X))
        rgb_bp.set_attribute('image_size_y', str(CAMERA_Y))
        rgb_bp.set_attribute('sensor_tick', str(1/POLL_RATE))
        
        transform = carla.Transform(carla.Location(x=0.60, y=0.25, z=1.8))
        r_rgb = world.world.spawn_actor(rgb_bp, transform, attach_to=world.player, attachment_type=carla.AttachmentType.Rigid)
        
        
        depth_bp = bp_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(CAMERA_X))
        depth_bp.set_attribute('image_size_y', str(CAMERA_Y))
        depth_bp.set_attribute('sensor_tick', str(1/POLL_RATE))
        
        transform = carla.Transform(carla.Location(x=0.60, y=-0.25, z=1.8))
        depth = world.world.spawn_actor(depth_bp, transform, attach_to=world.player, attachment_type=carla.AttachmentType.Rigid)
        
        
        l_rgb_callback = lambda image: threading.Thread(target = rgb_callback, args = (image, 'left_rgb', output_path, world, world.player.get_transform())).start()
        r_rgb_callback = lambda image: threading.Thread(target = rgb_callback, args = (image, 'right_rgb', output_path, None, None)).start()
        l_depth_callback = lambda image: threading.Thread(target = depth_callback, args = (image, 'left_depth', output_path)).start()
        
        l_rgb.listen(l_rgb_callback)
        r_rgb.listen(r_rgb_callback)
        depth.listen(l_depth_callback)
        
        
        sensors = [l_rgb, r_rgb, depth]

        
        return world, controller, display, hud, agent, traffic_manager, sensors, output_path
            
    except Exception as e:
        print('Something went wrong setting up the simulation episode:')
        
        raise(e)
        
    

def sim_episode(client, args, iteration_name, episode_name): # uses code from automatic_control.py and generate_traffic.py
    """
    Single simulation episode loop. This handles updating all the HUD information,
    collecting and saving sensor data, ticking the agent and, if needed,
    the world. Expects simulation world to already have npc's spawned and
    cameras set. Calls prep_episode() to spawn player and set cameras.
    """

    global position_dict
    
    world, controller, display, hud, agent, traffic_manager, sensors, output_path = prep_episode(client, args, iteration_name, episode_name)
    

    try:
        spawn_points = world.map.get_spawn_points()
        clock = pygame.time.Clock()
        num_ticks = 0
        first_tick = True
        s_lidar = None
        main_cam = sensors[0]
        
        global actors
        actors = world.world.get_actors()
        
        while num_ticks < 2000:
            clock.tick()
           
            if not args.asynch:
                
                position_dict[num_ticks] = main_cam.get_transform()
                if num_ticks >= 1:
                    if position_dict[num_ticks-1].location == position_dict[num_ticks].location:
                        #print('is still')
                        pass
                
                frame_num = world.world.get_snapshot().frame
                actor_bb_dict[frame_num] = {actor.id: (actor.bounding_box, actor.get_transform()) for actor in world.world.get_actors()}

                #print(position_dict.keys())
                world.world.tick()
                
                # Initialize semantic lidar on first tick
                if first_tick:
                    bp_library = world.world.get_blueprint_library()
                    
                    s_lidar_bp = bp_library.find('sensor.lidar.ray_cast_semantic')
                    s_lidar_bp.set_attribute('horizontal_fov', str(CAMERA_FOV))
                    s_lidar_bp.set_attribute('range', str(50.0))
                    s_lidar_bp.set_attribute('rotation_frequency', str(20.0))
                    s_lidar_bp.set_attribute('sensor_tick', str(1/POLL_RATE))
                    s_lidar_bp.set_attribute('lower_fov', str(-30.0))
                    s_lidar_bp.set_attribute('upper_fov', str(30.0))
                    s_lidar_bp.set_attribute('points_per_second', str(1792000))
                    s_lidar_bp.set_attribute('channels', str(256.0)) 
                    transform = carla.Transform(carla.Location(x=0.60, y=-0.25, z=1.8), carla.Rotation(pitch=0, yaw=0.0, roll=0.0))
                    s_lidar = world.world.spawn_actor(s_lidar_bp, transform, attach_to=world.player, attachment_type=carla.AttachmentType.Rigid)
                    
                    lid_callback = lambda data: threading.Thread(target = lidar_callback, args = (data, 'left_lidar', output_path, world.world.get_actors(), {actor.id: (actor.bounding_box, actor.get_transform()) for actor in world.world.get_actors()})).start()
                    s_lidar.listen(lid_callback)
                    
                    first_tick = False
                
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return
            
            #print('sim_time_now:', sim_time_now)
            
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
            
                
                
            if agent.done():
                agent.set_destination(random.choice(spawn_points).location)
                world.hud.notification("Target reached", seconds=4.0)
                print("The target has been reached, searching for another target")
                

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)
            num_ticks += 1

    finally:
        
        # Join all threads, not most eloquent solution but prevents simulation from ending while data is being processed in some cases
        for thread in threading.enumerate():
            if thread is not threading.currentThread():
                try:
                    thread.join()
                except:
                    None

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            
            for sensor in sensors:
                sensor.destroy()
                
            s_lidar.destroy()

            world.destroy()

        pygame.quit()
        
        # may or may not be needed for waiting to finish writes (idk)
        time.sleep(5)


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x400',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.nissan.patrol_2021',
        help='Actor model filter (default: "vehicle.nissan.patrol_2021")')
    argparser.add_argument(
        '--player-color',
        metavar='PLAYER_COLOR',
        default='150,150,150',
        help='Set a specific color for player car, like ["r, g, b"], where 255 is the max value')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=10,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--weather',
        action='store',
        default=carla.WeatherParameters.Default,
        help='Set weather preset')
    argparser.add_argument(
        '--map',
        action='store',
        type=str,
        default="Town01",
        help='Set map name')
    argparser.add_argument(
        '--external_behavior',
        action='store',
        choices=["cautious", "normal", "aggressive"],
        type=str,
        default="normal",
        help='Set behavior of external drivers')

    args = argparser.parse_args()
    
    #args.asynch = True
    
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    # List of weather presets
    weathers = [carla.WeatherParameters.Default, carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon, carla.WeatherParameters.WetNoon, carla.WeatherParameters.WetCloudyNoon, carla.WeatherParameters.MidRainyNoon, carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.SoftRainNoon, carla.WeatherParameters.ClearSunset, carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetSunset, carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.MidRainSunset, carla.WeatherParameters.HardRainSunset, carla.WeatherParameters.SoftRainSunset]
    
    # Construct iteration and episode name based on config
    iteration_name = ''
    iteration_name = iteration_name + EGO_BEHAVIOR
    iteration_name = iteration_name + '_w' + str(WEATHER)
    iteration_name = iteration_name + '_' + MAP
    
    episode_name = EGO_BEHAVIOR
    
    args.external_behavior = EXTERNAL_BEHAVIOR
    args.behavior = EGO_BEHAVIOR
    args.weather = weathers[WEATHER]
    args.map = MAP
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()
        client.load_world(args.map)

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)
        if args.weather:
            world.set_weather(args.weather)

        settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints = gt.get_actor_blueprints(world, args.filterv, args.generationv)
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = gt.get_actor_blueprints(world, args.filterw, args.generationw)
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        if args.safe:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
 
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)
                
        # Sets external vehicle behavior
        if args.external_behavior:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                if args.external_behavior == "aggressive":
                    traffic_manager.vehicle_percentage_speed_difference(actor, -30)
                    traffic_manager.distance_to_leading_vehicle(actor, 2)
                    traffic_manager.ignore_lights_percentage(actor, 60)
                    traffic_manager.ignore_signs_percentage(actor, 60)
                    traffic_manager.ignore_vehicles_percentage(actor, 60)
                    traffic_manager.ignore_walkers_percentage(actor, 60)
                    traffic_manager.random_left_lanechange_percentage(actor, 60)
                    traffic_manager.random_right_lanechange_percentage(actor, 60)
                if args.external_behavior == "normal":
                    traffic_manager.vehicle_percentage_speed_difference(actor, 0)
                    traffic_manager.distance_to_leading_vehicle(actor, 6)
                    traffic_manager.ignore_lights_percentage(actor, 30)
                    traffic_manager.ignore_signs_percentage(actor, 30)
                    traffic_manager.ignore_vehicles_percentage(actor, 30)
                    traffic_manager.ignore_walkers_percentage(actor, 30)
                    traffic_manager.random_left_lanechange_percentage(actor, 30)
                    traffic_manager.random_right_lanechange_percentage(actor, 30)
                if args.external_behavior == "cautious":
                    traffic_manager.vehicle_percentage_speed_difference(actor, 30)
                    traffic_manager.distance_to_leading_vehicle(actor, 10)
                    traffic_manager.ignore_lights_percentage(actor, 0)
                    traffic_manager.ignore_signs_percentage(actor, 0)
                    traffic_manager.ignore_vehicles_percentage(actor, 0)
                    traffic_manager.ignore_walkers_percentage(actor, 0)
                    traffic_manager.random_left_lanechange_percentage(actor, 0)
                    traffic_manager.random_right_lanechange_percentage(actor, 0)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()
            

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))
        
        
        
        try:
            sim_episode(client, args, iteration_name, episode_name)
        
        
        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
    finally:

        if hasattr(args, "asynch") and args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)
        
        if output_path:
            process_threads = []
            process_threads.append(threading.Thread(target=start_process, args=("left_rgb", l_rgb_dict, rgb_process, world)))
            process_threads.append(threading.Thread(target=start_process, args=("right_rgb", r_rgb_dict, rgb_process, world)))
            process_threads.append(threading.Thread(target=start_process, args=("depth", d_rgb_dict, depth_process, world)))
            process_threads.append(threading.Thread(target=start_process, args=("lidar", s_lidar_dict, lidar_process, world)))
            
            for thread in process_threads:
                thread.start()
            
            for thread in process_threads:
                thread.join()
            
            clean_data()
            write_episode_kitti(output_path)

def match_dynamic_static(dynamic_objects, static_objects, dynamic_tree_root):
    for dynamic_object in dynamic_objects:
        closest_static = (None, float('inf'))
        
        # Get center position of dynamic object
        dynamic_center_child = dynamic_object.find("center")
        dynamic_center = carla.Location(float(dynamic_center_child.attrib['x']), float(dynamic_center_child.attrib['y']), float(dynamic_center_child.attrib['z']))
        
        for static_object in static_objects:
            # Get center position of static object
            static_center_child = static_object.find("center")
            static_center = carla.Location(float(static_center_child.attrib['x']), float(static_center_child.attrib['y']), float(static_center_child.attrib['z']))
            
            # Compare to distance to previous closest static object
            cur_distance = abs(dynamic_center.distance(static_center))
            if cur_distance < closest_static[1]:
                closest_static = (static_object, cur_distance)
                
        # Assign dynamic actor id to closest static object
        if closest_static[0]:
            closest_static[0].set("actorId", dynamic_object.attrib["actorId"])
            object = closest_static[0]
            # Update dynamic object edges to closest static
            bbox_elem = ET.SubElement(dynamic_tree_root, 'BoundingBox', attrib=object.attrib)

            # copy all subelements of static obj into new dynamic actor xml element 
            for child in object:
                bbox_elem.append(child)
                        
            # mark previous box for comparison to new one
            if DEBUG_ON:
                dynamic_object.set('moved','True')
            else:
                dynamic_tree_root.remove(dynamic_object)

def move_unmatched_static(static_objects, static_tree, occlude_tree_root):
    for object in static_objects:
        # Objects without actorId are obscured
        if "actorId" not in object.attrib:
            bbox_elem = ET.SubElement(occlude_tree_root, 'BoundingBox', attrib=object.attrib)
            for child in object:
                child_copy = ET.SubElement(bbox_elem, child.tag, attrib=child.attrib)
                # Copy text content if any
                if child.text:
                    child_copy.text = child.text
        
        static_tree.remove(object)

def clean_data():
    frames = list(os.scandir(output_path))
    
    deleted_frames = 0
    
    for frame in frames:
        if frame.name == "config.ini":
            continue
        
        frame_data = os.scandir(frame.path)
        
        # Remove frames that are missing data
        if len(list(frame_data)) < FILES_PER_FRAME:
            shutil.rmtree(frame)
            deleted_frames += 1
            continue
        
        dynamic_tree = ET.parse(os.path.join(frame.path, 'dynamic_bbs.xml'))
        static_tree = ET.parse(os.path.join(frame.path, 'static_bbs.xml'))
        
        dynamic_tree_root = dynamic_tree.getroot()
        static_tree_root = static_tree.getroot()
        
        dynamic_traffic_lights = dynamic_tree_root.findall('.//BoundingBox[@class="TrafficLight"]')
        dynamic_traffic_signs = dynamic_tree_root.findall('.//BoundingBox[@class="TrafficSign"]')
        
        static_traffic_lights = static_tree_root.findall('.//BoundingBox[@class="TrafficLight"]')
        static_traffic_signs = static_tree_root.findall('.//BoundingBox[@class="TrafficSigns"]')
        
        # Match dynamic objects to static objects by distance
        match_dynamic_static(dynamic_traffic_lights, static_traffic_lights, dynamic_tree_root)
        match_dynamic_static(dynamic_traffic_signs, static_traffic_signs, dynamic_tree_root)
        
        # Create occluded bb tree and populate with occluded boxes
        occlude_tree_root = ET.Element("OccludedBoundingBoxes")
        occlude_tree = ET.ElementTree(occlude_tree_root)
        
        move_unmatched_static(static_traffic_lights, static_tree_root, occlude_tree_root)
        move_unmatched_static(static_traffic_signs, static_tree_root, occlude_tree_root)
        
        indent(occlude_tree_root)
        indent(dynamic_tree_root)
        
        static_tree.write(os.path.join(frame.path, 'static_bbs.xml'))
        occlude_tree.write(os.path.join(frame.path, 'obscured_bbs.xml'))
        dynamic_tree.write(os.path.join(frame.path, 'dynamic_bbs.xml'))
    
    frame_count = len(frames) - deleted_frames
    print("deleted frames:", deleted_frames)
    
    key = EGO_BEHAVIOR + "_" + EXTERNAL_BEHAVIOR + "_" + str(WEATHER) + "_" + MAP

    config = configparser.ConfigParser()
    config.read(os.path.join(DATA_PATH, "config.ini"))

    if 'FRAMECOUNTS' not in config:
        config['FRAMECOUNTS'] = {}
    if key in config['FRAMECOUNTS']:
        value = config['FRAMECOUNTS'][key]
        value = value + ',' + str(frame_count)
        config['FRAMECOUNTS'][key] = value
    else:
        config['FRAMECOUNTS'][key] = str(frame_count)

    with open(os.path.join(DATA_PATH, "config.ini"), 'w') as configfile:    # save
        config.write(configfile)
    
if __name__ == '__main__':
    main()