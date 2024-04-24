import tensorflow as tf


def eraser(img):
    """Random erasing as simple tensorflow function"""
    ## fixed
    erase_frac_lower = 0.05
    erase_frac_upper = 0.3
    erase_ratio = 0.3
    erase_value = 0
    ##
    img_shape = img.shape
    height = img_shape[0]
    width = img_shape[1]
    channels = img_shape[2] if len(img_shape) > 2 else 1
    area = height * width
    target_area = tf.random.uniform((), erase_frac_lower, erase_frac_upper) * area
    target_ratio = tf.random.uniform((), erase_ratio, 1/erase_ratio)
    target_height = int(tf.math.round(tf.math.sqrt(target_area) * target_ratio))
    target_width = int(tf.math.round(tf.math.sqrt(target_area) / target_ratio))
    print(target_height)
    if target_width < width and target_height < height:
        x = int(tf.random.uniform((), 0, width - target_width, dtype=tf.int32))
        y = int(tf.random.uniform((), 0, height - target_height, dtype=tf.int32))
        new_img = tf.Variable(img)
        if channels == 1:
            new_img[y:y + target_height, x:x + target_width].assign(erase_value)
        else:
            new_img[y:y + target_height, x:x + target_width, 0:channels].assign(erase_value)
        return tf.convert_to_tensor(new_img)
    return img
