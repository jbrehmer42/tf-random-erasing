import pytest

from erasing.limit_layer import ErasingLayerWithLimits


@pytest.mark.parametrize(
    ["limits"],
    [
        ((0.1, 0.1, 0.3),),
        ((0.1, 0.2, 0.2, 0.2, 0.3),),
        ((0.1, 0.0, -0.1, 0.2),),
        ((0.5, 0.0, -0.6, 0.2),),
        ((0.5, 0.1, 0.5, 0.0),),
        ((0.5, 0.1, 0.6, 0.0),),
        ((0.0, 0.3, 0.0, 0.7),),
        ((0.2, 0.5, 0.5, 0.5),),
        ((0, 0, 1, 0),),
    ],
)
def test_validate_limits_failures(limits):
    """Exception must be raised for invalid area limits"""
    layer = ErasingLayerWithLimits()
    with pytest.raises(ValueError):
        _ = layer.validate_limits(limits)


@pytest.mark.parametrize(
    ["limits"], [((0, 0, 0, 0),), ((0.1, 0, 0.2, 0),), ((0.5, 0.5, 0.0, 0),)]
)
def test_validate_limits_format(limits):
    """Limit values must be float after successful validation"""
    layer = ErasingLayerWithLimits()
    new_limits = layer.validate_limits(limits)
    assert all((isinstance(limit, float) for limit in new_limits))
    assert new_limits == limits
