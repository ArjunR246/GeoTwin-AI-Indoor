import pytest

from geotwin.floorplan.building_model import (
    BuildingModel,
)
from geotwin.floorplan.corridors import Corridor
from geotwin.floorplan.geometry3d import (
    Building3D,
    Corridor3D,
    Room3D,
    Wall3D,
    build_building_3d,
    polygon_from_object,
)
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


@pytest.fixture
def model(wall, room, corridor):
    return BuildingModel(
        image_width=1200,
        image_height=800,
        walls=(wall,),
        rooms=(room,),
        corridors=(corridor,),
    )


def test_wall3d_creation():
    wall = Wall3D(
        id="wall_1",
        start=(0.0, 0.0),
        end=(100.0, 0.0),
        height_m=3.2,
    )

    assert wall.id == "wall_1"
    assert wall.start == (0.0, 0.0)
    assert wall.end == (100.0, 0.0)
    assert wall.height_m == 3.2


def test_wall3d_length_uses_source_coordinate_units():
    wall = Wall3D(
        id="wall_1",
        start=(0.0, 0.0),
        end=(3.0, 4.0),
    )

    assert wall.length == pytest.approx(5.0)


def test_wall3d_bottom_vertices():
    wall = Wall3D(
        id="wall_1",
        start=(0.0, 0.0),
        end=(100.0, 0.0),
    )

    assert wall.bottom_vertices == (
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
    )


def test_wall3d_top_vertices():
    wall = Wall3D(
        id="wall_1",
        start=(0.0, 0.0),
        end=(100.0, 0.0),
        height_m=3.2,
    )

    assert wall.top_vertices == (
        (0.0, 0.0, 3.2),
        (100.0, 0.0, 3.2),
    )


def test_wall3d_surface_vertices():
    wall = Wall3D(
        id="wall_1",
        start=(0.0, 0.0),
        end=(100.0, 0.0),
        height_m=3.2,
    )

    assert wall.vertices == (
        (0.0, 0.0, 0.0),
        (100.0, 0.0, 0.0),
        (100.0, 0.0, 3.2),
        (0.0, 0.0, 3.2),
    )


def test_wall3d_accepts_optional_thickness_and_material():
    wall = Wall3D(
        id="wall_1",
        start=(0.0, 0.0),
        end=(10.0, 0.0),
        thickness_m=0.2,
        material="concrete",
    )

    assert wall.thickness_m == 0.2
    assert wall.material == "concrete"


@pytest.mark.parametrize("height", [0, -1, float("inf"), float("nan")])
def test_wall3d_rejects_invalid_height(height):
    with pytest.raises(ValueError):
        Wall3D(
            id="wall_1",
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            height_m=height,
        )


def test_wall3d_rejects_identical_endpoints():
    with pytest.raises(ValueError):
        Wall3D(
            id="wall_1",
            start=(1.0, 1.0),
            end=(1.0, 1.0),
        )


def test_wall3d_rejects_invalid_thickness():
    with pytest.raises(ValueError):
        Wall3D(
            id="wall_1",
            start=(0.0, 0.0),
            end=(10.0, 0.0),
            thickness_m=0,
        )


def test_room3d_creation(room):
    room3d = Room3D(
        id=room.id,
        polygon=polygon_from_object(room),
    )

    assert room3d.id == "room_1"
    assert len(room3d.polygon) == 4
    assert room3d.ceiling_height_m == 3.2


def test_room3d_floor_vertices(room):
    room3d = Room3D(
        id=room.id,
        polygon=polygon_from_object(room),
    )

    assert room3d.floor_vertices == (
        (10.0, 10.0, 0.0),
        (100.0, 10.0, 0.0),
        (100.0, 100.0, 0.0),
        (10.0, 100.0, 0.0),
    )


def test_room3d_ceiling_vertices(room):
    room3d = Room3D(
        id=room.id,
        polygon=polygon_from_object(room),
        ceiling_height_m=3.2,
    )

    assert room3d.ceiling_z_m == pytest.approx(3.2)
    assert all(
        vertex[2] == pytest.approx(3.2)
        for vertex in room3d.ceiling_vertices
    )


def test_room3d_wall_surface_count(room):
    room3d = Room3D(
        id=room.id,
        polygon=polygon_from_object(room),
    )

    assert len(room3d.wall_vertices) == 4
    assert all(len(surface) == 4 for surface in room3d.wall_vertices)


def test_room3d_supports_nonzero_floor_height(room):
    room3d = Room3D(
        id=room.id,
        polygon=polygon_from_object(room),
        floor_z_m=1.5,
        ceiling_height_m=3.2,
    )

    assert room3d.floor_vertices[0][2] == 1.5
    assert room3d.ceiling_z_m == pytest.approx(4.7)


def test_room3d_rejects_polygon_with_too_few_points():
    with pytest.raises(ValueError):
        Room3D(
            id="room_bad",
            polygon=((0.0, 0.0), (1.0, 1.0)),
        )


def test_room3d_rejects_invalid_ceiling_height():
    with pytest.raises(ValueError):
        Room3D(
            id="room_1",
            polygon=(
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
            ),
            ceiling_height_m=0,
        )


def test_corridor3d_creation(corridor):
    corridor3d = Corridor3D(
        id=corridor.id,
        polygon=polygon_from_object(corridor),
    )

    assert corridor3d.id == "corridor_1"
    assert len(corridor3d.polygon) == 4
    assert corridor3d.ceiling_z_m == pytest.approx(3.2)


def test_corridor3d_floor_and_ceiling_vertices(corridor):
    corridor3d = Corridor3D(
        id=corridor.id,
        polygon=polygon_from_object(corridor),
    )

    assert len(corridor3d.floor_vertices) == 4
    assert len(corridor3d.ceiling_vertices) == 4

    assert all(
        vertex[2] == pytest.approx(0.0)
        for vertex in corridor3d.floor_vertices
    )

    assert all(
        vertex[2] == pytest.approx(3.2)
        for vertex in corridor3d.ceiling_vertices
    )


def test_polygon_from_room_preserves_coordinates(room):
    polygon = polygon_from_object(room)

    assert polygon == (
        (10.0, 10.0),
        (100.0, 10.0),
        (100.0, 100.0),
        (10.0, 100.0),
    )


def test_polygon_from_corridor_preserves_coordinates(corridor):
    polygon = polygon_from_object(corridor)

    assert polygon == (
        (10.0, 200.0),
        (500.0, 200.0),
        (500.0, 300.0),
        (10.0, 300.0),
    )


def test_build_building_3d_converts_all_objects(model):
    building = build_building_3d(model)

    assert isinstance(building, Building3D)
    assert building.wall_count == 1
    assert building.room_count == 1
    assert building.corridor_count == 1


def test_build_building_3d_uses_default_ceiling_height(model):
    building = build_building_3d(model)

    assert building.ceiling_height_m == 3.2
    assert building.walls[0].height_m == 3.2
    assert building.rooms[0].ceiling_height_m == 3.2
    assert building.corridors[0].ceiling_height_m == 3.2


def test_build_building_3d_preserves_source_xy_coordinates(model):
    building = build_building_3d(model)

    assert building.walls[0].start == (0.0, 0.0)
    assert building.walls[0].end == (100.0, 0.0)

    assert building.rooms[0].polygon[0] == (10.0, 10.0)
    assert building.corridors[0].polygon[0] == (10.0, 200.0)


def test_build_building_3d_accepts_custom_floor_id(model):
    building = build_building_3d(
        model,
        floor_id="bmsce_third_floor",
    )

    assert building.floor_id == "bmsce_third_floor"


def test_build_building_3d_accepts_custom_ceiling_height(model):
    building = build_building_3d(
        model,
        ceiling_height_m=3.5,
    )

    assert building.ceiling_height_m == 3.5
    assert building.walls[0].height_m == 3.5
    assert building.rooms[0].ceiling_height_m == 3.5
    assert building.corridors[0].ceiling_height_m == 3.5


def test_build_building_3d_accepts_wall_thickness(model):
    building = build_building_3d(
        model,
        wall_thickness_m=0.2,
    )

    assert building.walls[0].thickness_m == 0.2


def test_build_building_3d_rejects_invalid_model():
    with pytest.raises(TypeError):
        build_building_3d(None)


def test_build_building_3d_rejects_invalid_ceiling_height(model):
    with pytest.raises(ValueError):
        build_building_3d(
            model,
            ceiling_height_m=0,
        )


def test_build_building_3d_rejects_invalid_wall_thickness(model):
    with pytest.raises(ValueError):
        build_building_3d(
            model,
            wall_thickness_m=-0.1,
        )


def test_build_building_3d_rejects_empty_floor_id(model):
    with pytest.raises(ValueError):
        build_building_3d(
            model,
            floor_id="",
        )


def test_build_building_3d_handles_empty_model():
    model = BuildingModel(
        image_width=100,
        image_height=100,
        walls=(),
        rooms=(),
        corridors=(),
    )

    building = build_building_3d(model)

    assert building.wall_count == 0
    assert building.room_count == 0
    assert building.corridor_count == 0


def test_building3d_rejects_lists_instead_of_tuples():
    with pytest.raises(TypeError):
        Building3D(
            floor_id="floor_3",
            ceiling_height_m=3.2,
            walls=[],
            rooms=(),
            corridors=(),
        )


def test_building3d_rejects_invalid_wall_objects():
    with pytest.raises(TypeError):
        Building3D(
            floor_id="floor_3",
            ceiling_height_m=3.2,
            walls=("invalid",),
            rooms=(),
            corridors=(),
        )