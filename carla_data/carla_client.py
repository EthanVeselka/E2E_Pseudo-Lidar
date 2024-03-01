import sys
import config
CARLA_PYTHON_PATH = config.CARLA_PYTHON_PATH

if CARLA_PYTHON_PATH not in sys.path:
    sys.path.insert(0,CARLA_PYTHON_PATH)
    
import carla
import automatic_control as ac # Taken from Carla's example code
import generate_traffic as gt # Taken from Carla's example code
import argparse
import logging
import pygame
import random
from agents.navigation.behavior_agent import BehaviorAgent 
from agents.navigation.basic_agent import BasicAgent  
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  
import time
import datetime
import os
import numpy as np
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET
import configparser
import threading

POLL_RATE = config.POLL_RATE
CAMERA_X = config.CAMERA_X
CAMERA_Y = config.CAMERA_Y
DATA_PATH = config.DATA_PATH
CAMERA_FOV = config.CAMERA_FOV

EXTERNAL_BEHAVIOR = config.EXTERNAL_BEHAVIOR
EGO_BEHAVIOR = config.EGO_BEHAVIOR
WEATHER = config.WEATHER
MAP = config.MAP


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
            


def save_boxes(world, name, sample_path, transform, player_transform):
    print(world.player.get_transform().location)
    # Create the XML structure
    root = ET.Element("StaticBoundingBoxes")
    tree = ET.ElementTree(root)
    #bbs = world.world.get_level_bbs(carla.CityObjectLabel.Car)
    
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
        ]

    
    world_2_camera = np.array(transform.get_inverse_matrix())
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
    K = build_projection_matrix(CAMERA_X, CAMERA_Y, CAMERA_FOV)
    bounding_box_set = []
    
    for obj in filters:
        new_bbs = world.world.get_level_bbs(obj)
        for bb in new_bbs:
            bounding_box_set.append((obj, bb))
    count = 0
    for label, bb in bounding_box_set:

        # Filter for distance from ego vehicle
        if bb.location.distance(player_transform.location) < 50:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the bounding box. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = player_transform.get_forward_vector()
            ray = bb.location - player_transform.location
            
            if forward_vec.dot(ray) > 1:
                # Cycle through the vertices
                bbox_elem = ET.SubElement(root, "BoundingBox")
                bbox_elem.set("class", str(label))
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                counter = 0
                
                #bbs1 = [(new_bb.x, new_bb.y, new_bb.z) for new_bb in bb.get_world_vertices(player_transform)]
                #bbs2 = [(new_bb.x, new_bb.y, new_bb.z) for new_bb in bb.get_world_vertices(carla.Transform())]
                
                #print('player_transform:', bbs1)
                #print('carla transform():', bbs2)
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
                count += 1
                if not (x_min > 0 and x_max < CAMERA_X and y_min > 0 and y_max < CAMERA_Y):
                    on_screen.set('on_screen', 'true')
                else:
                    on_screen.set('on_screen', 'false')
                
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
                    

    # Save the bounding boxes in the scene
    filename = 'static_bbs.xml'
    file_path = os.path.join(sample_path, filename)
    indent(root)
    tree.write(file_path)
                

def rgb_callback(data, name, episode_path, world, player_transform):
    sample_path = os.path.join(episode_path, str(data.frame))
    #print(sample_path)
    
    # Folder may already exist
    try:
        if not os.path.exists(sample_path):
            #print(sample_path)
            os.mkdir(sample_path)
    except:
        None
    finally:
        file_name = '%s.png' % name
        full_path = os.path.join(sample_path, file_name)
        data.save_to_disk(full_path)
        
        if name == 'left_rgb':
            save_boxes(world, name, sample_path, data.transform, player_transform)
    
def depth_callback(data, name, episode_path):
    sample_path = os.path.join(episode_path, str(data.frame))
    #print(sample_path)
    
    # Folder may already exist
    try:
        if not os.path.exists(sample_path):
            #print(sample_path)
            os.mkdir(sample_path)
    except:
        None
    finally:
        file_name = '%s.png' % name
        full_path = os.path.join(sample_path, file_name)
        data.save_to_disk(full_path, color_converter=carla.ColorConverter.Depth)
    
    
def lidar_callback(data, name, episode_path, actors):
    sample_path = os.path.join(episode_path, str(data.frame))
    
    try:
        if not os.path.exists(sample_path):
            #print(sample_path)
            os.mkdir(sample_path)
    except:
        None
    finally:
        file_name = '%s.ply' % name
        full_path = os.path.join(sample_path, file_name)
        data.save_to_disk(full_path)
        
        actor_set = set()
        
        # Get unique actors from lidar data
        for point in data:
            if point.object_idx:
                actor_set.add(point.object_idx)
        
        world_2_camera = np.array(data.transform.get_inverse_matrix())
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        K = build_projection_matrix(CAMERA_X, CAMERA_Y, CAMERA_FOV)
        
        root = ET.Element("DynamicBoundingBoxes")
        tree = ET.ElementTree(root)
        
        for actor_id in actor_set:
            actor = actors.find(actor_id)
            
            bbox_elem = ET.SubElement(root, "BoundingBox")
            bbox_elem.set("class", str(type(actor)))

            bb = actor.bounding_box
            verts = [v for v in bb.get_world_vertices(carla.Transform())]
            counter = 0
            for edge in edges:
                    # Join the vertices into edges
                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                    p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
                    # Draw the edges into the camera output
                    bbox_elem_edge = ET.SubElement(bbox_elem, "edge" + str(counter))
                    
                    bbox_elem_edge.set("x1", str(p1[0]))
                    bbox_elem_edge.set("y1", str(p1[1]))
                    bbox_elem_edge.set("x2", str(p2[0]))
                    bbox_elem_edge.set("y2", str(p2[1]))
                    
                    counter += 1
        
        # Save the bounding boxes in the scene
        filename = 'dynamic_bbs.xml'
        file_path = os.path.join(sample_path, filename)
        indent(root)
        tree.write(file_path)
    

    
 

def prep_episode(client, args, episode_name): # uses code from automatic_control.py and generate_traffic.py
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
        ts = datetime.datetime.now()
        data_path = os.path.join(DATA_PATH) 
        episode_path = os.path.join(data_path, episode_name)
        internal_path = os.path.join(episode_path, EGO_BEHAVIOR)
        output_path = os.path.join(internal_path, str(ts).replace(':', '-').replace('.', '-').replace(' ', '_'))
        
        if not os.path.exists(episode_path):
            os.mkdir(episode_path)
        
        if not os.path.exists(internal_path):
            os.mkdir(internal_path)
        
        os.mkdir(output_path)
        
        # Store config in data directories
        with open(os.path.join(episode_path, "episode_config.txt"), 'w') as dest:
            config = "External_behavior = " + EXTERNAL_BEHAVIOR + "\nMap = " + MAP + "\nWeather = " + str(WEATHER)
            dest.write(config)
        
        with open(os.path.join(output_path, "internal_config.txt"), 'w') as dest:
            config = "Ego_behavior = " + EGO_BEHAVIOR
            dest.write(config)
                
       
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
        
    

def sim_episode(client, args, episode_name): # uses code from automatic_control.py and generate_traffic.py
    """
    Single simulation episode loop. This handles updating all the HUD information,
    collecting and saving sensor data, ticking the agent and, if needed,
    the world. Expects simulation world to already have npc's spawned and
    cameras set. Calls prep_episode() to spawn player and set cameras.
    """

    world, controller, display, hud, agent, traffic_manager, sensors, output_path = prep_episode(client, args, episode_name)

    try:
        spawn_points = world.map.get_spawn_points()
        clock = pygame.time.Clock()
        num_ticks = 0
        first_tick = True
        s_lidar = None

        while num_ticks < 2000:
            clock.tick()
           
            if not args.asynch:
                world.world.tick()
                
                # Initialize semantic lidar on first tick
                if first_tick:
                    bp_library = world.world.get_blueprint_library()
                    
                    s_lidar_bp = bp_library.find('sensor.lidar.ray_cast_semantic')
                    s_lidar_bp.set_attribute('sensor_tick', str(1/POLL_RATE))
                    transform = carla.Transform(carla.Location(x=0.60, y=-0.25, z=1.8), carla.Rotation(pitch=180.0, yaw=0.0, roll=0.0))
                    s_lidar = world.world.spawn_actor(s_lidar_bp, transform, attach_to=world.player, attachment_type=carla.AttachmentType.Rigid)
                    
                    s_lidar.horizontal_fov = 90.0
                    s_lidar.range = 50.0
                    
                    lid_callback = lambda data: threading.Thread(target = lidar_callback, args = (data, 'left_lidar', output_path, world.world.get_actors())).start()
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

            world.destroy()

        pygame.quit()
        
        time.sleep(60)


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
        default='600x400',
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
    
    # Construct episode name based on config
    episode_name = ''
    
    args.behavior = EGO_BEHAVIOR
    
    args.external_behavior = EXTERNAL_BEHAVIOR
    episode_name = episode_name + EXTERNAL_BEHAVIOR

    args.weather = weathers[WEATHER]
    episode_name = episode_name + '_w' + str(WEATHER)

    args.map = MAP
    episode_name = episode_name + '_' + MAP
    
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
            sim_episode(client, args, episode_name)
        
        
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



if __name__ == '__main__':
    main()
    