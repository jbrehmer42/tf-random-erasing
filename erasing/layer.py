import tensorflow as tf
from typing import Any


class ErasingLayer(tf.keras.layers.Layer):
    """Random erasing layer. Randomly selects a fraction of the image area which
    should be erased. Then a rectangle with area of the given shape and random height
    and width is erased.
    """

    def __init__(
        self,
        erase_frac_lower: float = 0.05,
        erase_frac_upper: float = 0.1,
        erase_ratio: float = 0.3,
        name: str | None = None,
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
        """Try random erasing for a single image. Choose a random area, width, and
        height and erase the values in this area if it fits in the image. If not,
        return the image unchanged.
        """
        height = img.shape[0]
        width = img.shape[1]
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
            img = self.erase_target(img, x, y, target_height, target_width)
        return img

    @staticmethod
    def erase_target(
        img, x_loc: int, y_loc: int, target_height: int, target_width: int
    ):
        """Erase the target area specified by x, y location and height and width from
        the image.
        """
        channels = img.shape[2] if len(img.shape) > 2 else 1
        if channels == 1:
            indices = tf.stack(
                tf.meshgrid(
                    tf.range(y_loc, y_loc + target_height),
                    tf.range(x_loc, x_loc + target_width),
                ),
                axis=2,
            )
            updates = tf.zeros((target_width, target_height), dtype=tf.uint8)
            img = tf.tensor_scatter_nd_update(img, indices, updates)
        else:
            indices = tf.stack(
                tf.meshgrid(
                    tf.range(y_loc, y_loc + target_height),
                    tf.range(x_loc, x_loc + target_width),
                    tf.range(0, channels),
                ),
                axis=3,
            )
            updates = tf.zeros((target_width, target_height, channels), dtype=tf.uint8)
            img = tf.tensor_scatter_nd_update(img, indices, updates)
        return img

    def call(self, inputs, training: bool = True):
        if training:
            outputs = tf.cast(inputs, tf.uint8)
            return tf.map_fn(fn=self.erase_in_single_image, elems=outputs)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> dict[str, Any]:
        config = {
            "erase_frac_lower": self.erase_frac_lower,
            "erase_frac_upper": self.erase_frac_upper,
            "erase_ratio": self.erase_ratio,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
