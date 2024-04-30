''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import pickle
import cv2
from math import tan, pi, acos
import xml.etree.ElementTree as ET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "models"))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import provider
from train_util import get_batch

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log_v2/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
# parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
# parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
parser.add_argument('--draw_predictions', action='store_true', default=False, help='If true, draws predicted bounding boxes onto images')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 4

# Load Frustum Datasets.
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    rotate_to_center=True, overwritten_data_path=FLAGS.data_path,
    from_rgb_detection=False, one_hot=True)

def get_image_point(loc):
    # Calculate 2D projection of 3D coordinate
    Cx = 1280 / 2
    Cy = 720 / 2
    fov = 90

    f = 1280 /(2.0 * tan(fov * pi / 360))
    K = np.array([[f, 0, Cx], [0, f, Cy], [0, 0, 1]], dtype=np.float64)

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc[0], loc[1], loc[2]])

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def get_session_and_ops(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                is_training_pl)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}
        return sess, ops

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0]%batch_size == 0
    num_batches = int(pc.shape[0]/batch_size)
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],)) # 3D box score 
   
    ep = ops['end_points'] 
    for i in range(num_batches):
        feed_dict = {\
            ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
            ops['one_hot_vec_pl']: one_hot_vec[i*batch_size:(i+1)*batch_size,:],
            ops['is_training_pl']: False}

        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals']],
                feed_dict=feed_dict)

        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
        centers[i*batch_size:(i+1)*batch_size,...] = batch_centers
        heading_logits[i*batch_size:(i+1)*batch_size,...] = batch_heading_scores
        heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        size_logits[i*batch_size:(i+1)*batch_size,...] = batch_size_scores
        size_residuals[i*batch_size:(i+1)*batch_size,...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i*batch_size:(i+1)*batch_size] = batch_scores 
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
        size_cls, size_res, scores

def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list, \
                            path_list, obj_idx_list):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    
    obj_idx_dict = {}
    
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)
        
        # if path_list[i] not in obj_idx_dict:
            # obj_idx_dict[path_list[i]] = {}

        # Get pred 3d box
        box_size = provider.class2size(size_cls_list[i], size_res_list[i])
        heading_angle = provider.class2angle(
            heading_cls_list[i], heading_res_list[i], NUM_HEADING_BIN
        )
        corners_3d = provider.get_3d_box(box_size, heading_angle, center_list[i])

        img_corners_3d = []

        for point in corners_3d:
            img_point = get_image_point(point)
            img_corners_3d.append(img_point)
        
        if not np.isnan(score):
            if path_list[i] not in obj_idx_dict:
                obj_idx_dict[path_list[i]] = {}
            obj_idx_dict[path_list[i]][obj_idx_list[i]] = [output_str, img_corners_3d, score]

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close()
    
    for key, value in obj_idx_dict.items():
        path = key[0]
        output_path = os.path.join(path, "output")
        if os.path.exists(output_path):
            file_path = os.path.join(output_path, "labels.txt")
            if os.path.exists(file_path):
                os.remove(file_path)
        
    
    for key, value in obj_idx_dict.items():
        path = key[0]
        output_path = os.path.join(path, "output")
        if not os.path.exists(output_path): os.mkdir(output_path)
        file_path = os.path.join(output_path, "labels.txt")
        with open(file_path, 'a+') as label_file:
            for label in value.values():
                label_file.write(label[0]+'\n')
    
    if FLAGS.draw_predictions:
        # Sort paths by average frame score
        average_scores = {}
        for key, sub_dict in obj_idx_dict.items():
            path = key
            total_score = 0
            count = 0
            for idx_values in sub_dict.values():
                total_score += idx_values[2]
                count += 1
            # average_scores[path] = total_score / count if count > 0 else 0
            average_scores[path] = total_score
        sorted_paths = sorted(average_scores, key=average_scores.get, reverse=False)
        
        # Draw 10 best scoring frames
        idx = 0
        for path in sorted_paths[:50]:
            sub_dict = obj_idx_dict[path]
            image_path = os.path.join(path, "left_rgb.png")
            output_path = os.path.join(path, "output")
            label_path = os.path.join(output_path, "labels.txt")

            image = cv2.imread(image_path)
            
            # Draw ground truth boxes
            bb_path = os.path.join(path, "dynamic_bbs.xml")
            tree = ET.parse(bb_path)
            root = tree.getroot()
            
            for bb in root:
                for child in bb:
                    title = child.tag
                    if title.startswith('edge'):
                        start_pos = (int(float(child.attrib['x1'])), int(float(child.attrib['y1'])))
                        end_pos = (int(float(child.attrib['x2'])), int(float(child.attrib['y2'])))
                        cv2.line(image, start_pos, end_pos, (0, 255, 0), 2)
            
            # Draw prediction boxes
            for label in sub_dict.values():
                label_data = label[0].split()
                obj_class = label_data[0]
                edges = [[0,1], [1,2], [3,2], [3,0], [0,4], [4,5], [5,1], [5,6], [7,6], [7,4], [6,2], [7,3]]
                verts = label[1]
                for edge in edges:
                    p1 = verts[edge[0]]
                    p2 = verts[edge[1]]
                    cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
                x_min = int(float(verts[7][0]))
                y_min = int(float(verts[7][1]))
                cv2.putText(image, obj_class, (x_min-5, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                
            output_dir = os.path.join(result_dir, 'data')
            pred_image_path = os.path.join(output_dir, "prediction" + str(idx) + ".png")
            cv2.imwrite(pred_image_path, image)
            idx += 1    

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

def test_from_rgb_detection(output_filename, result_dir=None):
    ''' Test frustum pointents with 2D boxes from a RGB detector.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    onehot_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_size = BATCH_SIZE
    num_batches = int((len(TEST_DATASET)+batch_size-1)/batch_size)
    
    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx

        batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL, from_rgb_detection=True)
        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec

        # Run one batch inference
        batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
        inference(sess, ops, batch_data_to_feed, batch_one_hot_to_feed, batch_size=batch_size)
               
        for i in range(cur_batch_size):
            ps_list.append(batch_data[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            #score_list.append(batch_scores[i])
            score_list.append(batch_rgb_prob[i]) # 2D RGB detection score
            onehot_list.append(batch_one_hot_vec[i])

    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)
            pickle.dump(onehot_list, fp)

    # Write detection results for KITTI evaluation
    print('Number of point clouds: %d' % (len(ps_list)))
    write_detection_results(result_dir, TEST_DATASET.id_list,
        TEST_DATASET.type_list, TEST_DATASET.box2d_list,
        center_list, heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)
    # Make sure for each frame (no matter if we have measurement for that frame),
    # there is a TXT file
    output_dir = os.path.join(result_dir, 'data')
    if FLAGS.idx_path is not None:
        to_fill_filename_list = [line.rstrip()+'.txt' \
            for line in open(FLAGS.idx_path)]
        fill_files(output_dir, to_fill_filename_list)

def test(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    seg_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_size = BATCH_SIZE
    num_batches = int(len(TEST_DATASET)/batch_size)

    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
    correct_cnt = 0
    print("num_batches:", num_batches)
    print("len(TEST_DATASET)", len(TEST_DATASET))
    print("batch_size:", batch_size)
    for batch_idx in range((num_batches)):
        start_idx = batch_idx * batch_size
        
        end_idx = (batch_idx+1) * batch_size

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)

        batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data,
                batch_one_hot_vec, batch_size=batch_size)

        correct_cnt += np.sum(batch_output==batch_label)
        
        for i in range(batch_output.shape[0]):
            ps_list.append(batch_data[i,...])
            seg_list.append(batch_label[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])

    print("correct_cnt:", correct_cnt)
    print("float(batch_size*num_batches*NUM_POINT):", float(batch_size*num_batches*NUM_POINT))
    print("Segmentation accuracy: %f" % \
        (correct_cnt / float(batch_size*num_batches*NUM_POINT)))

    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(seg_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)

    # Write detection results for KITTI evaluation
    write_detection_results(result_dir, TEST_DATASET.id_list,
        TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list,
        heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list,
        TEST_DATASET.path_list, TEST_DATASET.obj_idx_list)


if __name__=='__main__':
    # if FLAGS.from_rgb_detection:
    #     test_from_rgb_detection(FLAGS.output+'.pickle', FLAGS.output)
    # else:
    test(FLAGS.output+'.pickle', FLAGS.output)
