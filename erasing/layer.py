import tensorflow as tf


class ErasingLayer(tf.keras.layers.Layer):
    """Random erasing layer. Randomly selects a fraction of the image area which
    should be erased. Then a rectangle with area of the given shape and random height
    and width is erased.
    """

    def __init__(
        self,
        erase_frac_lower=0.05,
        erase_frac_upper=0.1,
        erase_ratio=0.3,
        name=None,
        **kwargs
    ):
        """Create erasing layer

        :param erase_frac_lower: Lower limit of the fraction of image area to erase
        :param erase_frac_upper: Upper limit of the fraction of image area to erase
        :param erase_ratio: Limit of the aspect ratio of the rectangle to erase
        """
        self.erase_frac_lower = erase_frac_lower
        self.erase_frac_upper = erase_frac_upper
        self.erase_ratio = erase_ratio
        super().__init__(name=name, **kwargs)

    def erase_in_single_image(self, img):
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        channels = img_shape[2] if len(img_shape) > 2 else 1
        area = height * width
        target_area = (
            tf.random.uniform((), self.erase_frac_lower, self.erase_frac_upper) * area
        )
        target_ratio = tf.random.uniform((), self.erase_ratio, 1 / self.erase_ratio)
        target_height = int(tf.math.round(tf.math.sqrt(target_area) * target_ratio))
        target_width = int(tf.math.round(tf.math.sqrt(target_area) / target_ratio))
        if target_width < width and target_height < height:
            x = int(tf.random.uniform((), 0, width - target_width, dtype=tf.int32))
            y = int(tf.random.uniform((), 0, height - target_height, dtype=tf.int32))
            if channels == 1:
                indices = tf.stack(
                    tf.meshgrid(
                        tf.range(y, y + target_height), tf.range(x, x + target_width)
                    ),
                    axis=2,
                )
                updates = tf.zeros((target_width, target_height), dtype=tf.uint8)
                img = tf.tensor_scatter_nd_update(img, indices, updates)
            else:
                indices = tf.stack(
                    tf.meshgrid(
                        tf.range(y, y + target_height),
                        tf.range(x, x + target_width),
                        tf.range(0, 3),
                    ),
                    axis=3,
                )
                updates = tf.zeros((target_width, target_height, 3), dtype=tf.uint8)
                img = tf.tensor_scatter_nd_update(img, indices, updates)
        return img

    def call(self, inputs):
        outputs = tf.cast(inputs, tf.uint8)
        # Todo: Make this work for unbatched images?
        return tf.map_fn(fn=self.erase_in_single_image, elems=outputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "erase_frac_lower": self.erase_frac_lower,
            "erase_frac_upper": self.erase_frac_upper,
            "erase_ratio": self.erase_ratio,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
