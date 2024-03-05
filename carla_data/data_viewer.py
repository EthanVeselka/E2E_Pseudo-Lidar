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

conf = configparser.ConfigParser()
conf.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))

CARLA_PYTHON_PATH = conf["Paths"]["CARLA_PYTHON_PATH"]
DATA_PATH = conf["Paths"]["DATA_PATH"]

if CARLA_PYTHON_PATH not in sys.path:
    sys.path.insert(0,CARLA_PYTHON_PATH)

POLL_RATE = float(conf["Settings"]["POLL_RATE"])
CAMERA_X = int(conf["Settings"]["CAMERA_X"])
CAMERA_Y = int(conf["Settings"]["CAMERA_Y"])
CAMERA_FOV = int(conf["Settings"]["CAMERA_FOV"])

EGO_BEHAVIOR = conf["Internal Variables"]["EGO_BEHAVIOR"]
EXTERNAL_BEHAVIOR = conf["External Variables"]["EXTERNAL_BEHAVIOR"]
WEATHER = int(conf["External Variables"]["WEATHER"])
MAP = conf["External Variables"]["MAP"]

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

def load_bbs(sample_path):
    bbs = []
    
    bb_types = set()
    
    try:
        filename1 = 'dynamic_bbs.xml'
        path1 = os.path.join(sample_path, filename1)
        tree1 = ET.parse(path1)
        
        filename2 = 'static_bbs.xml'
        path2 = os.path.join(sample_path, filename2)
        tree2 = ET.parse(path2)
        
        root1 = tree1.getroot()
        root2 = tree2.getroot()
        
        for child in root1:
            bb_types.add(child.attrib['class'])
            bbs.append(child)
    
        
        for child in root2:
            #print(child.tag, child.attrib)
            bb_types.add(child.attrib['class'])
            #print(child.attrib['class'])
            bbs.append(child)
    except:
        None
    
    
    return bb_types, bbs
    
    

def view_image(display, image_path, frame_num, bbs, bb_class, font):
    imp = pygame.image.load(image_path).convert()
    
    
    for bb in bbs:
        #print('bb:', bb)
        #print('bb_stuff:', bb.tag, bb.attrib)
        if bb.attrib['class'] == bb_class:
            for child in bb:
                title = child.tag
                if title.startswith('edge'):
                    #print('startpos:', (child.attrib['x1'], child.attrib['y1']))
                    start_pos = (float(child.attrib['x1']), float(child.attrib['y1']))
                    end_pos = (float(child.attrib['x2']), float(child.attrib['y2']))
                    pygame.draw.line(imp, (255,0,0), start_pos, end_pos)

    

            
    # Using blit to copy content from one surface to other
    text_img = font.render(str(bb_class), True, (255, 0, 0))
    display.blit(imp, (0, 0))
    pygame.display.flip()
    display.blit(text_img, (0, 0))

    #print(str(bb_class))
    
    
    
    
    
    
    


def main():
    
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
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
    
    # Construct episode name based on config
    
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
    
    pygame.init()
    pygame.font.init()
    #print(pygame.font.get_fonts())
    font = pygame.font.SysFont('arial', 48)


    
    display = pygame.display.set_mode(
        (args.width, args.height),
        pygame.HWSURFACE | pygame.DOUBLEBUF)

    
    data_path = os.path.join(DATA_PATH) 
    episode_path = os.path.join(data_path, episode_name)
    internal_path = os.path.join(episode_path, iteration_name)
    
    iterations = os.listdir(internal_path)
    
    #print(iterations[-1])
    
    iteration_path = os.path.join(internal_path, iterations[-1])
    samples = os.listdir(iteration_path)
    sorted_samples = []
    for sample in samples:
        try:
            sorted_samples.append(str(int(sample)))
        except:
            pass
        
    sorted_samples.sort()
    samples = sorted_samples
    #print(samples)
    first_sample = samples[0]
    first_sample_path = os.path.join(iteration_path, first_sample)
    left_rgb_path = os.path.join(first_sample_path, 'left_rgb.png')
    
    bb_classes, bbs = load_bbs(first_sample_path)
    for sample in samples:
        try:
            sample_path = os.path.join(iteration_path, sample)
            new_classes, _ = load_bbs(sample_path)
            bb_classes = bb_classes.union(new_classes)
            #print(bb_classes)
        except Exception as e:
            print(e) 
    bb_classes = list(bb_classes)
    bb_class_i = 0

    print(bb_classes)
    imp = pygame.image.load(left_rgb_path).convert()
 
    # Using blit to copy content from one surface to other
    display.blit(imp, (0, 0))
     
    # paint screen one time
    
    #box_types = ['dynamic', 'poll']
    
    
    
    
    
    pygame.display.flip()
    status = True
    
    sample_i = 0
    max_sample_i = len(samples) - 1
    
    img_types = ['left_rgb.png', 'right_rgb.png',] #'left_depth.png']
    img_type_i = 0
    max_img_type_i = len(img_types)
    
    show_bb = False
    
    l_hold = float('inf')
    r_hold = float('inf')
    while (status):
     
      # iterate over the list of Event objects
      # that was returned by pygame.event.get() method.
      
        for event in pygame.event.get():
     
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                status = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    sample_i -= 1
                    sample_i %= max_sample_i
                    l_hold = time.perf_counter()
                if event.key == pygame.K_RIGHT:
                    sample_i += 1
                    sample_i %= max_sample_i
                    r_hold = time.perf_counter()
                if event.key == pygame.K_DOWN:
                    img_type_i -= 1
                    img_type_i %= max_img_type_i
                if event.key == pygame.K_UP:
                    img_type_i += 1
                    img_type_i %= max_img_type_i
                if event.key == pygame.K_q:
                    bb_class_i -= 1
                    bb_class_i %= len(bb_classes)-1
                if event.key == pygame.K_e:
                    bb_class_i += 1
                    bb_class_i %= len(bb_classes)-1
                if event.key == pygame.K_SPACE:
                    if show_bb:
                        show_bb = False
                    else:
                        show_bb = True
                print(bb_classes[bb_class_i])
                
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    l_hold = float('inf')
                if event.key == pygame.K_RIGHT:
                    r_hold = float('inf')
                    
            
                
        now = time.perf_counter()
        if now - l_hold > 0.5:
            sample_i -= 1
            sample_i %= max_sample_i
        if now - r_hold > 0.5:
            sample_i += 1
            sample_i %= max_sample_i
            
        sample = samples[sample_i]
        sample_path = os.path.join(iteration_path, sample)
        img_name = img_types[img_type_i]
        image_path = os.path.join(sample_path, img_name)
        _, bbs = load_bbs(sample_path)
        bb_class = bb_classes[bb_class_i]     
        try:
            view_image(display, image_path, sample, bbs, bb_class, font)
        except:
            None
        pygame.display.update()
        
            
        #print(r_hold)
                
    # deactivates the pygame library
    pygame.quit()
    
    
    
    
    


if __name__ == '__main__':
    main()