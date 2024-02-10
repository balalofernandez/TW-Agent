from .Headquarters import Headquarters
from .Academy import Academy
from .Barracks import Barracks
from .Clay import Clay
from .Church import Church
from .Farm import Farm
from .First_church import First_church
from .Hiding_place import Hiding_place
from .Iron import Iron
from .Market import Market
from .Rally_point import Rally_point
from .Smithy import Smithy
from .Stable import Stable
from .Statue import Statue
from .Timber import Timber
from .Wall import Wall
from .Warehouse import Warehouse
from .Watchtower import Watchtower
from .Workshop import Workshop

building_registry = {
    "headquarters": Headquarters,
    "academy": Academy,
    "barracks": Barracks,
    "clay": Clay,
    "church": Church,
    "farm": Farm,
    "first_church": First_church,
    "hiding_place": Hiding_place,
    "iron": Iron,
    "market": Market,
    "rally_point": Rally_point,
    "smithy": Smithy,
    "stable": Stable,
    "statue": Statue,
    "timber": Timber,
    "wall": Wall,
    "warehouse": Warehouse,
    "watchtower": Watchtower,
    "workshop": Workshop,
}

def create_building(building_name,level=0,*args,**kwargs):
    building_class = building_registry.get(building_name)
    if building_class:
        return building_class(level,*args,**kwargs)  # Instantiate the building
    else:
        raise ValueError(f"Unknown building type: {building_name}")