from PIL import Image
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt

example_dir = '/Users/seancondon/Desktop/065_final/dataset-medium/images/2ef3a4994a_0CCD105428INSPIRE-ortho.tif'
processed_dir = '/Users/seancondon/Desktop/065_final/ingested_images/'
STRIDE_LEN = 256

def im_to_array(indir):
    '''
    Converts a .tif image into a numpy array.

    :param indir: Directory specifying the .tif image to convert.
    :return: 3-channel np array.
    '''
    #open image from directory using PIL
    im = Image.open(indir)
    ar = np.array(im)

    #chop off fourth channel from image
    ar2 = ar[:,:,:3]

    #optional imshow to make sure image looks okay
    if False:
        out_im = Image.fromarray(ar2)
        out_im.show()

    return ar2


def ingest_array(ar):
    '''
    Convert a large 3-channel np array into small square chunks of size STRIDE_LEN x STRIDE_LEN.
    Ignore such chunks that contain a majority black pixels.

    :param ar: Large 3-channel np array to ingest.
    :return: TODO
    '''
    #declare stride length and get image dimensions
    h, w, channels = ar.shape

    #clip off edges of image
    h_len = h // STRIDE_LEN
    h_overflow = h % STRIDE_LEN
    w_len = w // STRIDE_LEN
    w_overflow = w % STRIDE_LEN
    cropped_ar = ar[:-h_overflow, :-w_overflow,:]


    #split array into square chunks
    out = []
    h_split = np.split(cropped_ar, h_len, axis = 0)
    for h_splitted in h_split:
        w_splitted = np.split(h_splitted, w_len, axis = 1)
        for i in w_splitted:
            out.append(i)
    out_ar = np.array(out)

    #loop through out_ar, discard chunks with > 50% black pixels
    THRESHOLD = 0.5
    to_exp = []
    total_pix = (STRIDE_LEN**2)*3
    for i in range(out_ar.shape[0]):
        arr = out_ar[i]
        zer_count = np.count_nonzero(arr==0)
        zer_freq = zer_count / total_pix
        if zer_freq <= THRESHOLD:
            to_exp.append(arr)

    return np.array(to_exp)

def handle_meta_image(file_dir):
    '''
    Handle the ingestion for a single .tif image, first coverting the image to a numpy array and
    then chunking it.

    :param file_dir: directory for tif image to ingest
    :return: TODO
    '''
    print('Ingesting %s...'%(file_dir.split('/')[-1]))

    #convert tif to np array
    im_array = im_to_array(file_dir)

    #chunk image twice, so we have slices offset by STRIDE_LEN / 2
    half_stride = STRIDE_LEN / 2
    assert half_stride == int(half_stride)
    half_stride = int(half_stride)
    non_offset_meta = ingest_array(im_array)
    offset_meta = ingest_array(im_array[half_stride:,half_stride:,:])

    #vstack both arrays
    out_array = np.concatenate((non_offset_meta, offset_meta), axis=0)

    #save ingested chunks
    outdir = os.path.join(processed_dir, file_dir.split('/')[-1].split('.')[0])
    try:
        os.makedirs(outdir)
    except:
        pass

    for i in range(out_array.shape[0]):
        tosave = out_array[i]
        #tosave = np.divide(tosave, 255)
        filename = '%d.npy'%(i)
        np.save(file=os.path.join(outdir, filename), arr=tosave)
    print('Done Ingesting!')
    print('Output shape = '+str(out_array.shape))

def examine_output():
    directory = '/Users/seancondon/Desktop/065_final/ingested_images/c644f91210_27E21B7F30OPENPIPELINE-ortho/'
    toShow = 0
    for f in os.listdir(directory):
        #if f.endswith('.npy'):
        if f == '191.npy':
            im_array = np.load(os.path.join(directory, f))
            #im_array = np.multiply(im_array, 255)
            print(im_array)
            im = Image.fromarray(im_array)
            im.show()
            toShow -=1
            if toShow < 0: break

def process_all(meta_dir):

    for f in os.listdir(meta_dir):
        if f.endswith('.tif'):
            handle_meta_image(os.path.join(meta_dir, f))


def tile_plot():
    dir = '/Users/seancondon/Desktop/065_final/ingested_images/'
    toPlot = 25
    images = []
    meta = os.listdir(dir)
    new_meta = []
    for f in meta:
        if f != '.DS_Store':
            new_meta.append(f)
    meta = new_meta
    for i in range(toPlot):
        meta_choice = random.choice(meta)
        im_choice = '.DS_Store'
        while not im_choice.endswith('.npy'):
            im_choice = random.choice(os.listdir(os.path.join(dir, meta_choice)))
        im_dir = os.path.join(dir, meta_choice)
        im_dir = os.path.join(im_dir, im_choice)
        im_arr = np.load(im_dir)
        im = Image.fromarray(im_arr)
        images.append(im)

    #show all images in image list
    fig = plt.figure(figsize=(6, 6))  # width, height in inches
    plt.tight_layout()
    #fig.subplots_adjust(hspace=0.1)
    for i in range(toPlot):
        sub = fig.add_subplot(5, 5, i + 1)
        sub.get_yaxis().set_visible(False)
        sub.get_xaxis().set_visible(False)
        sub.imshow(images[i], interpolation='nearest')
    fig.suptitle('25 Example Tiles', fontsize=16)
    plt.savefig('/Users/seancondon/Desktop/065_final/tiles.png',dpi=300)



tile_plot()
#process_all('/Users/seancondon/Desktop/065_final/dataset-medium/images/')
#examine_output()



