import tensorflow as tf
from typing import Any

from erasing.base_layer import ErasingBase


class ErasingLayer(ErasingBase):
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
        """Create an erasing layer

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
        target_height = tf.cast(
            tf.math.round(tf.math.sqrt(target_area) * target_ratio), tf.int32
        )
        target_width = tf.cast(
            tf.math.round(tf.math.sqrt(target_area) / target_ratio), tf.int32
        )
        if target_width < width and target_height < height:
            x = tf.random.uniform((), 0, width - target_width, dtype=tf.int32)
            y = tf.random.uniform((), 0, height - target_height, dtype=tf.int32)
            img = self.erase_target(img, x, y, target_height, target_width)
        return img

    def get_config(self) -> dict[str, Any]:
        config = {
            "erase_frac_lower": self.erase_frac_lower,
            "erase_frac_upper": self.erase_frac_upper,
            "erase_ratio": self.erase_ratio,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
