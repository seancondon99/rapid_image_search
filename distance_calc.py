#imports
import numpy as np
import operator
import json
import os
import time
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
import math


def save_global_dictionary():
    '''
    Aggregates all .json files in ./fingerprints to turn them into one global json dictionary

    :return: None, output saved as .json file
    '''

    #loop through all .json files in ./fingerprints
    global_dict = {}
    global_dir = './fingerprints/'
    for f in os.listdir(global_dir):
        if f.endswith('.json'):
            fname = f.split('_')[0]
            fpath = os.path.join(global_dir, f)
            with open(fpath) as fi:
                fi_dict = json.load(fi)
        for key, value in fi_dict.items():
            global_key = fname + '_' + key
            global_dict[global_key] = value

    savedir = '/Users/seancondon/Desktop/065_final/global.json'
    with open(savedir, 'w+') as fx:
        json.dump(global_dict, fx)

def shorten_img_name(original_string):
    #  generate UID from image original name by grabbing characters before first underscore
    short_name = original_string.split('_', 1)[0]
    return short_name


# transforms dict for img with uid = "test" from {"0" -> [512 floats], ...} to {"test_0 -> [512 floats], ...}
def concat_uid_with_tile_label(fp_folder_filepath, filename, uid):
    full_filepath = os.path.join(fp_folder_filepath, filename)
    # print(full_filepath)

    with open(full_filepath) as f:
        data = json.load(f)

    concat_dict = {f'{uid}_{k}': v for k, v in data.items()}
    global_image_dict.update(concat_dict)


def create_candidate_dict():
    fp_folder_filepath = '/Users/noaschwartz/Downloads/finalproject_065/fingerprints'  # same as root

    # list of json file names
    img_file_names = [img_json for img_json in os.listdir(fp_folder_filepath) if img_json.endswith('.json')]

    # walk through the directory
    for filename in img_file_names:

        short_name = shorten_img_name(filename)

        if short_name != filename:
            # rename img file name with new short uid (ignore file ext)
            filename_no_extension, extension = os.path.splitext(filename)

            path1 = os.path.join(fp_folder_filepath, filename)  # old path
            path2 = os.path.join(fp_folder_filepath, filename_no_extension.replace(filename_no_extension, short_name) + extension)

            # rename file with uid
            os.rename(path1, path2)
            # print("file: " + path1 + "  renamed to: " + path2)

        uid = os.path.splitext(short_name)[0]  # removes .json
        # print(uid)
        concat_uid_with_tile_label(fp_folder_filepath,short_name,uid)


def distance_score_between_fingerprint(query_fingerpint, candidate_fingerprint):
    dist = np.linalg.norm(query_fingerpint - candidate_fingerprint)
    return dist

def nearest_k_tiles(query_tile_uid, candidate_dict, k):
    query_fingerprint = candidate_dict[query_tile_uid]
    cand_dist_dict = {}  # maps uid of tile to fingerprint distance

    for cand_tile in candidate_dict.items():
        cand_uid, cand_fp = cand_tile
        dist = distance_score_between_fingerprint(np.array(query_fingerprint), np.array(cand_fp))
        cand_dist_dict[cand_uid] = dist

    # find tile uids with k smallest distance
    smallest_k_cand = dict(sorted(cand_dist_dict.items(), key=operator.itemgetter(1))[:k])
    return smallest_k_cand.keys()

def matplot_images(images, title = False):
    nrows, ncols = 1 , len(images)
    f, axarr = plt.subplots(nrows=nrows, ncols=ncols)

    squished = False
    for i in range(len(images)):
        if i > 0 and not squished:
            squished = True
            f.subplots_adjust(wspace = 0.1)
        axarr[i].get_yaxis().set_visible(False)
        axarr[i].get_xaxis().set_visible(False)
        plt.sca(axarr[i]);
        plt.imshow(images[i]);
        if i == 0:
            if title: plt.title('query')
        else:
            if title: plt.title('%d' % (i))
    plt.show()
    return f, axarr

def manage_query(q = None, K = 11):

    #load in global dict
    print('Loading global dict...')
    t0 = time.time()
    global_dir = '/Users/seancondon/Desktop/065_final/global.json'
    with open(global_dir) as gdir:
        global_dict = json.load(gdir)
    t1 = time.time()
    print('Dict load took %.4f seconds'%(t1-t0))
    keylist = list(global_dict.keys())

    #select query image, either using q if provided in function call, or picking a random query
    if q == None:
        QUERY = random.choice(keylist)
    else:
        QUERY = q

    #get k nearest tiles
    print('Finding nearest tiles...')
    t2 = time.time()
    nearest_tiles = nearest_k_tiles(QUERY, global_dict, K)
    t3 = time.time()
    print('Nearest tiles took %.4f seconds'%(t3-t2))

    #PLOTTING FUNCTIONALITY TO SEE RESULTS
    #imshow query image
    image_dir = './ingested_images/'
    query_image_path = os.path.join(image_dir, '%s_*'%(QUERY.split('_')[0]))
    query_image_path = os.path.join(query_image_path, '%s.npy'%(QUERY.split('_')[1]))
    resolved_q_path = glob.glob(query_image_path)
    q_im_arr = np.load(resolved_q_path[0])
    q_im = Image.fromarray(q_im_arr)
    images = [q_im]

    #imshow nearest_tiles
    for tile in nearest_tiles:
        image_path = os.path.join(image_dir, '%s_*' % (tile.split('_')[0]))
        image_path = os.path.join(image_path, '%s.npy' % (tile.split('_')[1]))
        resolved_im_path = glob.glob(image_path)
        im_arr = np.load(resolved_im_path[0])
        im = Image.fromarray(im_arr)
        if resolved_im_path != resolved_q_path:
            images.append(im)

    f, axarr = matplot_images(images)
    return f, axarr




manage_query()


#print(nearest_k_tiles('1d4fbe33f3_25', global_image_dict, 6))


good_query = ['ec09336a6f_250', #trees / barren land
              '1476907971_558', #more barren land (no trees)
              '15efe45820_3757', #petit houses w cars
              '107f24d6e9_2798', #tree shadows on short grass
              '39e77bedd0_480', #solar panels
              'f9f43e5144_514', #riverside in winter w trees,
              '107f24d6e9_2578', #nice houses w pool
              ]