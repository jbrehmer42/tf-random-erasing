import tensorflow as tf

from erasing.base_layer import ErasingBase


class ErasingLayerWithLimits(ErasingBase):
    """Random erasing layer which limits the erasing to a rectangle area of the image.
    Only parts of this are can be deleted, the rest of the image is 'protected'.
    """

    def __init__(
        self,
        erase_frac_lower: float = 0.05,
        erase_frac_upper: float = 0.1,
        erase_ratio: float = 0.3,
        name: str | None = None,
        area_limits: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        **kwargs
    ):
        """Create erasing layer with limits

        :param erase_frac_lower: Lower limit of the fraction of image area to erase.
            Relative to the selected erasing area.
        :param erase_frac_upper: Upper limit of the fraction of image area to erase
            Relative to the selected erasing area.
        :param erase_ratio: Limit of the aspect ratio of the rectangle to erase
        :param area_limits: Area to select for random erasing. Values are fractions of
            the image's width/height which define the limits relative to top, right,
            bottom, and left. The default is (0, 0, 0, 0), representing the whole
            image. To select, e.g. the upper right quarter of the image choose
            (0, 0, 0.5, 0.5).
        """
        self.erase_frac_lower = erase_frac_lower
        self.erase_frac_upper = erase_frac_upper
        self.erase_ratio = erase_ratio
        if any(area_limits):
            area_limits = self.validate_limits(area_limits)
        self.top, self.right, self.bottom, self.left = area_limits
        super().__init__(name=name, **kwargs)

    @staticmethod
    def validate_limits(limits: tuple[float, ...]) -> tuple[float, float, float, float]:
        """Check whether the selected limits are valid"""
        if len(limits) != 4:
            raise ValueError(
                f"Length of the specified area limits is {len(limits)}, but must be "
                f"4. One value for each of top, right, bottom, and left."
            )
        elif any(limit < 0 for limit in limits):
            raise ValueError(f"Negative limits found in area limits {limits}")
        elif limits[0] + limits[2] >= 1 - 1e-8:
            raise ValueError(
                f"Invalid limits. Sum of top limit (={limits[0]}) and bottom limit "
                f"(={limits[2]}) must not add up to 1."
            )
        elif limits[1] + limits[3] >= 1 - 1e-8:
            raise ValueError(
                f"Invalid limits. Sum of right limit (={limits[1]}) and left limit "
                f"(={limits[3]}) must not add up to 1."
            )
        return tuple(float(limit) for limit in limits)  # noqa

    def get_erasing_position(
        self, width: int, height: int, target_width: tf.Tensor, target_height: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Get x and y coordinates of the rectangle to delete. Pixel data are ordered
        from left to right and top to bottom

        :param width: Image width
        :param height: Image height
        :param target_width: Width of the rectangle targeted for erasing
        :param target_height: Height of the rectangle targeted for erasing
        """
        x_lim_left = tf.cast(tf.math.ceil(self.left * width), tf.int32)
        x_lim_right = tf.cast(tf.math.floor((1 - self.right) * width), tf.int32)
        y_lim_top = tf.cast(tf.math.ceil(self.top * height), tf.int32)
        y_lim_bottom = tf.cast(tf.math.ceil((1 - self.bottom) * height), tf.int32)
        x = tf.random.uniform(
            (), x_lim_left, x_lim_right - target_width, dtype=tf.int32
        )
        y = tf.random.uniform(
            (), y_lim_top, y_lim_bottom - target_height, dtype=tf.int32
        )
        return x, y

    def erase_in_single_image(self, img):
        """Try random erasing for a single image. Choose a random rectangle and erase
        the values in this rectangle if it fits in the selected erasing area. If not,
        return the image unchanged.
        """
        height = img.shape[0]
        width = img.shape[1]
        red_height = int(height * (1 - self.top - self.bottom))
        red_width = int(width * (1 - self.right - self.left))
        area = red_height * red_width
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
        if target_width < red_width - 1 and target_height < red_height - 1:
            x, y = self.get_erasing_position(width, height, target_width, target_height)
            img = self.erase_target(img, x, y, target_height, target_width)
        return img
