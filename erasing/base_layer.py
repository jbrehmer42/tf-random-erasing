import tensorflow as tf


class ErasingBase(tf.keras.layers.Layer):
    """Base class for all erasing layers. Provides call and erase_target method.
    Children must specify which area should be deleted via erase_in_single_image method

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
        **Note**: `"channels_first"` format not supported yet.

    Output shape: same as input shape
    """

    @staticmethod
    def erase_target(img, x_loc, y_loc, target_height, target_width):
        """Erase the target area specified by x, y location and height and width from
        the image.
        """
        channels = img.shape[2]
        indices = tf.stack(
            tf.meshgrid(
                tf.range(y_loc, y_loc + target_height),
                tf.range(x_loc, x_loc + target_width),
                tf.range(0, channels),
            ),
            axis=3,
        )
        updates = tf.zeros((target_width, target_height, channels), dtype=tf.uint8)
        return tf.tensor_scatter_nd_update(img, indices, updates)

    def call(self, inputs, training: bool = True):
        if training:
            unbatched = len(inputs.shape) == 3
            if unbatched:
                inputs = tf.expand_dims(inputs, axis=0)
                outputs = tf.cast(inputs, tf.uint8)
                outputs = tf.map_fn(fn=self.erase_in_single_image, elems=outputs)
                return tf.squeeze(outputs, axis=0)
            outputs = tf.cast(inputs, tf.uint8)
            return tf.map_fn(fn=self.erase_in_single_image, elems=outputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape
