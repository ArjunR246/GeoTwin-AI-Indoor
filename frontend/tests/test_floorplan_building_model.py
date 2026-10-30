import pytest

from geotwin.floorplan.building_model import (
    BuildingModel,
    build_building_model,
)
from geotwin.floorplan.corridors import Corridor
from geotwin.floorplan.rooms import Room
from geotwin.floorplan.walls import WallSegment


@pytest.fixture
def wall():
    return WallSegment(
        x1=0,
        y1=0,
        x2=100,
        y2=0,
        orientation="horizontal",
        length=100.0,
    )


@pytest.fixture
def room():
    return Room(
        id="room_1",
        polygon=(
            (10, 10),
            (100, 10),
            (100, 100),
            (10, 100),
        ),
        area=8100.0,
        centroid=(55.0, 55.0),
        bounding_box=(10, 10, 90, 90),
        aspect_ratio=1.0,
    )


@pytest.fixture
def corridor():
    return Corridor(
        id="corridor_1",
        polygon=(
            (10, 200),
            (500, 200),
            (500, 300),
            (10, 300),
        ),
        area=49000.0,
        centroid=(255.0, 250.0),
        bounding_box=(10, 200, 490, 100),
        aspect_ratio=4.9,
    )


def test_building_model_creation(wall, room, corridor):
    model = BuildingModel(
        image_width=1200,
        image_height=800,
        walls=(wall,),
        rooms=(room,),
        corridors=(corridor,),
    )

    assert model.image_width == 1200
    assert model.image_height == 800
    assert model.walls == (wall,)
    assert model.rooms == (room,)
    assert model.corridors == (corridor,)


def test_building_model_counts(wall, room, corridor):
    model = BuildingModel(
        image_width=1200,
        image_height=800,
        walls=(wall,),
        rooms=(room,),
        corridors=(corridor,),
    )

    assert model.wall_count == 1
    assert model.room_count == 1
    assert model.corridor_count == 1


def test_image_shape(wall, room, corridor):
    model = BuildingModel(
        image_width=1200,
        image_height=800,
        walls=(wall,),
        rooms=(room,),
        corridors=(corridor,),
    )

    assert model.image_shape == (800, 1200)


def test_total_room_area(wall, room, corridor):
    model = BuildingModel(
        image_width=1200,
        image_height=800,
        walls=(wall,),
        rooms=(room,),
        corridors=(corridor,),
    )

    assert model.total_room_area == 8100.0


def test_total_corridor_area(wall, room, corridor):
    model = BuildingModel(
        image_width=1200,
        image_height=800,
        walls=(wall,),
        rooms=(room,),
        corridors=(corridor,),
    )

    assert model.total_corridor_area == 49000.0


def test_empty_building_model_is_allowed():
    model = BuildingModel(
        image_width=100,
        image_height=100,
        walls=(),
        rooms=(),
        corridors=(),
    )

    assert model.wall_count == 0
    assert model.room_count == 0
    assert model.corridor_count == 0


def test_invalid_width_is_rejected():
    with pytest.raises(ValueError):
        BuildingModel(
            image_width=0,
            image_height=100,
            walls=(),
            rooms=(),
            corridors=(),
        )


def test_invalid_height_is_rejected():
    with pytest.raises(ValueError):
        BuildingModel(
            image_width=100,
            image_height=0,
            walls=(),
            rooms=(),
            corridors=(),
        )


def test_walls_must_be_tuple():
    with pytest.raises(TypeError):
        BuildingModel(
            image_width=100,
            image_height=100,
            walls=[],
            rooms=(),
            corridors=(),
        )


def test_rooms_must_be_tuple():
    with pytest.raises(TypeError):
        BuildingModel(
            image_width=100,
            image_height=100,
            walls=(),
            rooms=[],
            corridors=(),
        )


def test_corridors_must_be_tuple():
    with pytest.raises(TypeError):
        BuildingModel(
            image_width=100,
            image_height=100,
            walls=(),
            rooms=(),
            corridors=[],
        )


def test_invalid_wall_object_is_rejected():
    with pytest.raises(TypeError):
        BuildingModel(
            image_width=100,
            image_height=100,
            walls=("invalid",),
            rooms=(),
            corridors=(),
        )


def test_invalid_room_object_is_rejected():
    with pytest.raises(TypeError):
        BuildingModel(
            image_width=100,
            image_height=100,
            walls=(),
            rooms=("invalid",),
            corridors=(),
        )


def test_invalid_corridor_object_is_rejected():
    with pytest.raises(TypeError):
        BuildingModel(
            image_width=100,
            image_height=100,
            walls=(),
            rooms=(),
            corridors=("invalid",),
        )


def test_build_building_model_converts_lists(
    wall,
    room,
    corridor,
):
    model = build_building_model(
        image_width=1200,
        image_height=800,
        walls=[wall],
        rooms=[room],
        corridors=[corridor],
    )

    assert isinstance(model, BuildingModel)
    assert isinstance(model.walls, tuple)
    assert isinstance(model.rooms, tuple)
    assert isinstance(model.corridors, tuple)


def test_build_building_model_preserves_objects(
    wall,
    room,
    corridor,
):
    model = build_building_model(
        image_width=1200,
        image_height=800,
        walls=[wall],
        rooms=[room],
        corridors=[corridor],
    )

    assert model.walls[0] is wall
    assert model.rooms[0] is room
    assert model.corridors[0] is corridor