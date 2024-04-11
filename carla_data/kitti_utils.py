import xml.etree.ElementTree as ET
import os
from math import pi
import configparser

def write_episode_kitti(output_path):
    # declare class names from written classes in CARLA data saves
    class_names = {
        'carla.libcarla.Vehicle' : 'Vehicle',
        'carla.libcarla.Pedestrian' : 'Pedestrian',
        'carla.libcarla.TrafficSign' : 'TrafficSign',
        'carla.libcarla.TrafficLight' : 'TrafficLight',
        'Car' : 'Car',
        'Truck' : 'Truck',
        'Van' : 'Van',
        'Bus' : 'Bus',
        'Bicycle' : 'Bicycle',
        'Motorcycle' : 'Motorcycle',
        'TrafficLight' : 'TrafficLight',
        'Vehicle' : 'Vehicle',
        'TrafficSign' : 'TrafficSign',
        'Pedestrian' : 'Pedestrian'
    }

    # go through each sample in the specified output_path
    for dir in os.listdir(output_path):
        # skip if the 'dir' is actually a config file. (IDK why this is pulled in os.listdir())
        if dir == 'config.ini':
            continue
        sample_path = os.path.join(output_path, dir)
        dynamic_boxes_path = os.path.join(sample_path, 'dynamic_bbs.xml')

        # attempt to read boxes data in the sample
        boxes_data = None
        try:
            boxes_data = ET.parse(dynamic_boxes_path)
        except:
            continue
        kitti_data_path = os.path.join(sample_path, 'labels.txt')

        # go through each bounding box in the xml file
        root = boxes_data.getroot()
        with open(kitti_data_path, 'w') as dest:
            for child in root:
                bb_class = child.get('class')
                if bb_class in class_names:
                    # get desired attrbutes of the bounding boxes
                    label = class_names[bb_class]
                    orientation = child.find('relative_orientation')
                    Box2d = child.find('Box2d')
                    x_min = Box2d.attrib['xMin']
                    x_max = Box2d.attrib['xMax']
                    y_min = Box2d.attrib['yMin']
                    y_max = Box2d.attrib['yMax']
                    shape = child.find('shape')
                    height = shape.attrib['z']
                    width = shape.attrib['y']
                    length = shape.attrib['x']
                    center = child.find('relative_center')
                    x = center.attrib['x']
                    y = center.attrib['y']
                    z = center.attrib['z']
                    rot_y = float(orientation.attrib['yaw']) * pi / 180

                    # put desired attributes in string
                    alpha_str = orientation.get('observation_angle') + ' '
                    box2d_str = x_min + ' ' + y_min + ' ' + x_max + ' ' + y_max + ' '
                    shape_str = height + ' ' + width + ' ' + length + ' '
                    pos_str = x + ' ' + y + ' ' + z + ' '
                    rot_str = str(rot_y) + '\n'

                    # format all attribute strings into final string
                    out_str = label + ' 1 0 ' + alpha_str + box2d_str + shape_str + pos_str + rot_str
                    # write string to sample kitti output file
                    dest.write(out_str)
                else:
                    print('unexpected class name:', bb_class)

# if main, do write_episode_kitti() for path specified in config file
if __name__ == '__main__':
    # directory pathing to find most recent path with values specified by config file
    conf = configparser.ConfigParser()
    conf.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
    data_path = conf["Paths"]["DATA_PATH"]

    episode_name = conf["Internal Variables"]["EGO_BEHAVIOR"]
    episode_path = os.path.join(data_path, episode_name)

    iteration_name = ''
    iteration_name = iteration_name + conf["Internal Variables"]["EGO_BEHAVIOR"]
    iteration_name = iteration_name + '_w' + conf["External Variables"]["WEATHER"]
    iteration_name = iteration_name + '_' + conf["External Variables"]["MAP"]
    iteration_path = os.path.join(episode_path, iteration_name)

    # find most recently modified directory in iteration path
    most_recent_path = None
    recent_time = 0
    for dir in os.listdir(iteration_path):
        output_path = os.path.join(iteration_path, dir)
        if recent_time < os.path.getmtime(output_path):
            most_recent_path = output_path
            recent_time =  os.path.getmtime(output_path)

    print('output path:', most_recent_path)
    # create kitti data for that run
    write_episode_kitti(most_recent_path)




 

    
