import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes
from dataset.imdb import flip_boxes
from utils.image_processing import color_transform

classes = ['__background__',  # always index 0
           'airplane', 'antelope', 'bear', 'bicycle',
           'bird', 'bus', 'car', 'cattle',
           'dog', 'domestic_cat', 'elephant', 'fox',
           'giant_panda', 'hamster', 'horse', 'lion',
           'lizard', 'monkey', 'motorcycle', 'rabbit',
           'red_panda', 'sheep', 'snake', 'squirrel',
           'tiger', 'train', 'turtle', 'watercraft',
           'whale', 'zebra']
classes_map = ['__background__',  # always index 0
               'n02691156', 'n02419796', 'n02131653', 'n02834778',
               'n01503061', 'n02924116', 'n02958343', 'n02402425',
               'n02084071', 'n02121808', 'n02503517', 'n02118333',
               'n02510455', 'n02342885', 'n02374451', 'n02129165',
               'n01674464', 'n02484322', 'n03790512', 'n02324045',
               'n02509815', 'n02411705', 'n01726692', 'n02355227',
               'n02129604', 'n04468005', 'n01662784', 'n04530566',
               'n02062744', 'n02391049']

num_classes = len(classes)

_roidb_file = []

_im_tfile = []


def imread_from_tar(filepath, frame, flag):
    global _im_tfile
    import tarfile

    exist_flag = False
    for i in range(len(_im_tfile)):
        if _im_tfile[i]['path'] == filepath:
            tarf = _im_tfile[i]['tarfile']
            exist_flag = True
            break

    if not exist_flag:
        tarf = tarfile.open(filepath)
        _im_tfile.append({
            'path': filepath,
            'tarfile': tarf
        })

    name = './%010d.jpg' % frame
    if name in tarf.getnames():
        data = np.asarray(bytearray(tarf.extractfile(name).read()), dtype=np.uint8)
        return cv2.imdecode(data, flag)
    else:
        return None


# TODO: This two functions should be merged with individual data loader
def get_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        if 'tar' in roi_rec['pattern']:
            assert os.path.exists(roi_rec['pattern']), '%s does not exist'.format(roi_rec['pattern'])
            im = imread_from_tar(roi_rec['pattern'], roi_rec['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
            im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb


def get_pair_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0  # 0 for unequal, 1 for equal
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            ref_id = min(
                max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET + 1),
                    0), roi_rec['frame_seg_len'] - 1)
            ref_image = roi_rec['pattern'] % ref_id
            assert os.path.exists(ref_image), '%s does not exist'.format(ref_image)
            ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if ref_id == roi_rec['frame_seg_id']:
                eq_flag = 1
        else:
            ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb


def load_vid_annotation(image_path, data_path):
    """
    for a given index, load image and bounding boxes info from XML file
    :param index: index of a specific image
    :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    """

    import xml.etree.ElementTree as ET
    roi_rec = dict()
    filename = image_path.replace('JPEG', 'xml')
    filename = filename.replace('Data', 'Annotations')
    tree = ET.parse(filename)
    size = tree.find('size')
    roi_rec['height'] = float(size.find('height').text)
    roi_rec['width'] = float(size.find('width').text)

    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
    valid_objs = np.zeros((num_objs), dtype=np.bool)

    class_to_index = dict(zip(classes_map, range(num_classes)))
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = np.maximum(float(bbox.find('xmin').text), 0)
        y1 = np.maximum(float(bbox.find('ymin').text), 0)
        x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width'] - 1)
        y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height'] - 1)
        if not class_to_index.has_key(obj.find('name').text):
            continue
        valid_objs[ix] = True
        cls = class_to_index[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    boxes = boxes[valid_objs, :]
    gt_classes = gt_classes[valid_objs]
    overlaps = overlaps[valid_objs, :]

    assert (boxes[:, 2] >= boxes[:, 0]).all()

    roi_rec.update({'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'max_classes': overlaps.argmax(axis=1),
                    'max_overlaps': overlaps.max(axis=1),
                    'flipped': False})
    return roi_rec


def img_aug(name, im, config, new_rec, origin_size, max_size, shape_diff=False):
    from augmentations import SSDAugmentation
    if new_rec[name].shape[0] != 0:
        label = np.ones((new_rec[name].shape[0],), dtype=bool)
        im_aug, bbs_aug, mask = SSDAugmentation(mean=config.network.PIXEL_MEANS,
                                                expand_scale=config.TRAIN.ssd_expand_scale,
                                                crop_pert=config.TRAIN.ssd_crop_pert,
                                                color=config.TRAIN.ssd_color,
                                                no_iou_limit=config.TRAIN.ssd_no_iou_limit,
                                                )(im.copy(), new_rec[name].copy(), label)
        im_aug = im_aug.astype(np.uint8)
        if shape_diff:
            im_aug, im_scale = resize_to_2(im_aug, origin_size[0], origin_size[1], stride=config.network.IMAGE_STRIDE)
            tmp_im = np.zeros((origin_size[0], origin_size[1], 3),
                              dtype=im_aug.dtype)
            tmp_im[:, :, :] = config.network.PIXEL_MEANS
            tmp_im[0:im_aug.shape[0], 0:im_aug.shape[1], :] = im_aug
            im_aug = tmp_im
            im_scale = [im_scale, im_scale]
        else:
            im_aug, im_scale = resize_to(im_aug, origin_size, max_size, \
                                         stride=config.network.IMAGE_STRIDE)
    else:
        bbs_aug = None
        if shape_diff:
            im, im_scale = resize_to_2(im, origin_size[0], origin_size[1], stride=config.network.IMAGE_STRIDE)
            tmp_im = np.zeros((origin_size[0], origin_size[1], 3),
                              dtype=im.dtype)
            tmp_im[:, :, :] = config.network.PIXEL_MEANS
            tmp_im[0:im.shape[0], 0:im.shape[1], :] = im
            im = tmp_im
            im_scale = [im_scale, im_scale]
        else:
            im, im_scale = resize_to(im, origin_size, max_size, stride=config.network.IMAGE_STRIDE)

    if bbs_aug is not None:
        bbs_aug[:, :4] *= [im_scale[0], im_scale[1], im_scale[0], im_scale[1]]

    # need to have at lease one boxes if used for loss
    if config.TRAIN.loss_frames != 1:
        if bbs_aug is not None:
            if bbs_aug.shape[0] != 0:
                new_rec[name] = bbs_aug.copy()
            else:
                new_rec[name] = np.zeros((1, 4))
            new_rec[name.split('_')[0] + '_gt_classes'] = new_rec[name.split('_')[0] + '_gt_classes'][mask]

    if bbs_aug is not None:
        im = im_aug

    return im, bbs_aug


def get_triple_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_bef_ims = []
    processed_aft_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        shape_diff = False
        if roi_rec.has_key('pattern'):
            # get two different frames from the interval [frame_id + MIN_OFFSET, frame_id + MAX_OFFSET]
            offsets = np.random.choice(config.TRAIN.MAX_OFFSET - config.TRAIN.MIN_OFFSET + 1, 2, replace=False) \
                      + config.TRAIN.MIN_OFFSET

            bef_id = min(max(roi_rec['frame_seg_id'] + offsets[0], 0), roi_rec['frame_seg_len'] - 1)
            aft_id = min(max(roi_rec['frame_seg_id'] + offsets[1], 0), roi_rec['frame_seg_len'] - 1)
            bef_image = roi_rec['pattern'] % bef_id
            aft_image = roi_rec['pattern'] % aft_id

            assert os.path.exists(bef_image), '%s does not exist'.format(bef_image)
            assert os.path.exists(aft_image), '%s does not exist'.format(aft_image)
            bef_im = cv2.imread(bef_image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            aft_im = cv2.imread(aft_image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            if config.TRAIN.data_aug:
                data_path = config.dataset.dataset_path
                bef_rec = load_vid_annotation(bef_image, data_path)
                aft_rec = load_vid_annotation(aft_image, data_path)

            condition_has_flipped = False
        else:
            # for DET, conditional frames' gt boxes are the same as the training frame,
            # thus also flipped gt boxes if flip
            # only gt boxes has been flipped, not image
            condition_has_flipped = True
            bef_im = im.copy()
            aft_im = im.copy()

            if config.TRAIN.data_aug:
                bef_rec = roi_rec.copy()
                aft_rec = roi_rec.copy()

        # gt bbox if flipped outside this when the flip flag is True
        if roidb[i]['flipped']:
            # have to flip the training image since the gt boxes is already flipped
            im = im[:, ::-1, :]
            # do random flip on the conditional frames
            # be careful with the gt boxes if conditional frames are also included in loss
            # or using ssd-like augmentation, need the gt boxes correct for cropping the image
            if config.TRAIN.random_flip_condition_frames:
                flip_flags = np.random.choice(2, config.TRAIN.train_frames - 1, replace=True)
                if flip_flags[0]:
                    bef_im = bef_im[:, ::-1, :]
                if flip_flags[1]:
                    aft_im = aft_im[:, ::-1, :]
            else:
                flip_flags = np.ones(config.TRAIN.train_frames - 1)
                bef_im = bef_im[:, ::-1, :]
                aft_im = aft_im[:, ::-1, :]

            if condition_has_flipped:
                flip_flags = 1 - flip_flags

        else:
            flip_flags = np.zeros(config.TRAIN.train_frames - 1)

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)

        # data augmentation color change
        if config.TRAIN.data_aug:
            color_factor = config.TRAIN.COLOR_FACTOR
            im = color_transform(im, color_factor)
            bef_im = color_transform(bef_im, color_factor)
            aft_im = color_transform(aft_im, color_factor)

        # bbox transformation corresponding to image resize operation
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        if config.TRAIN.data_aug and not config.TRAIN.noncur_aug:
            from augmentations import SSDAugmentation
            label = np.ones((new_rec['boxes'].shape[0],), dtype=bool)
            im_aug, bbs_aug, mask = SSDAugmentation(mean=config.network.PIXEL_MEANS,
                                                    expand_scale=config.TRAIN.ssd_expand_scale,
                                                    crop_pert=config.TRAIN.ssd_crop_pert,
                                                    color=config.TRAIN.ssd_color
                                                    )(im.copy(), new_rec['boxes'].copy(), label)
            im_aug = im_aug.astype(np.uint8)
            im_aug, _im_scale = resize(im_aug, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            bbs_aug = bbs_aug * _im_scale

            if np.sum(mask) != 0:
                im = im_aug
                if bbs_aug.shape[0] != 0:
                    new_rec['boxes'] = bbs_aug.copy()
                else:
                    new_rec['boxes'] = np.zeros((1, 4))
                new_rec['gt_classes'] = new_rec['gt_classes'][mask]

        train_img_size = im.shape[:2]

        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]

        if config.TRAIN.data_aug:
            # correct label for conditional frames if their losses is used.
            if flip_flags[0]:
                old_boxes = bef_rec['boxes'].copy()
                flipped_boxes = flip_boxes(old_boxes, bef_rec['width'])
                bef_rec['boxes'] = flipped_boxes
            if flip_flags[1]:
                old_boxes = aft_rec['boxes'].copy()
                flipped_boxes = flip_boxes(old_boxes, aft_rec['width'])
                aft_rec['boxes'] = flipped_boxes

            new_rec['bef_gt_classes'] = bef_rec['gt_classes'].copy()
            new_rec['aft_gt_classes'] = aft_rec['gt_classes'].copy()

        bef_im, bef_im_scale = resize(bef_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        aft_im, aft_im_scale = resize(aft_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)

        if config.TRAIN.data_aug:
            new_rec['bef_boxes'] = bef_rec['boxes'].copy() * bef_im_scale
            new_rec['aft_boxes'] = aft_rec['boxes'].copy() * aft_im_scale

        if config.TRAIN.data_aug:
            origin_size = train_img_size
            bef_im, bef_bbs_aug = img_aug('bef_boxes', bef_im, config, new_rec, origin_size, max_size, shape_diff)
            aft_im, aft_bbs_aug = img_aug('aft_boxes', aft_im, config, new_rec, origin_size, max_size, shape_diff)

        bef_im_tensor = transform(bef_im, config.network.PIXEL_MEANS)
        aft_im_tensor = transform(aft_im, config.network.PIXEL_MEANS)
        processed_bef_ims.append(bef_im_tensor)
        processed_aft_ims.append(aft_im_tensor)

        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)

    return processed_ims, processed_bef_ims, processed_aft_ims, processed_roidb


def resize(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale


def resize_to(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape[:2]
    im_scale = np.zeros((2,), dtype=np.float32)
    im_scale[1] = float(target_size[0]) / float(im_shape[0])
    im_scale[0] = float(target_size[1]) / float(im_shape[1])
    # prevent bigger axis from being more than max_size:
    # if np.round(im_scale * im_size_max) > max_size:
    #     im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, (target_size[1], target_size[0]), interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale


def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor


def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor


def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im


def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind * islice:(ind + 1) * islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor


def resize_to_2(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = im_shape[0]
    im_size_max = im_shape[1]
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale
