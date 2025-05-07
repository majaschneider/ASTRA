from typing import Iterable, Iterator
import tomli
from pathlib import Path

def read_poi_descriptions(paths: Iterable[Path]) -> dict[str, dict[str, dict[str, str]]]:
    poi_descriptions = dict()
    for path in paths:
        with path.open("rb") as f:
            poi_descriptions.update(tomli.load(f))
    return poi_descriptions

def poi_transform(poi_descriptions: dict[str, dict[str, dict[str, str]]]) -> Iterable[str]:
    for cat, sections in poi_descriptions.items():
        for section, values in sections.items():
            if isinstance(values, dict):
                for value, description in values.items():
                    if value == "yes":
                        yield f"At/In {cat}: {description}"
                    if cat == "shop":
                        yield f"At/In shop ({section.lower()}). {value}: {description}".replace(
                            "_", " "
                        )
                    elif section == "other":
                        yield f"At/In {value}: {description}".replace("_", " ")
                    else:
                        yield f"At place for {section.lower()}. {value}: {description}".replace(
                            "_", " "
                        )
            else:
                if values == "yes":
                    yield f"At/In {cat}: {values}"
                elif cat == "office":
                    yield f"At office ({section.lower()}): {values}".replace("_", " ")
                else:
                    yield f"At/in {section.lower()}: {values}".replace("_", " ")

class CellEmptyException(Exception):
    """Raise this when a cell does not contain POIs, but they are required."""



# def transform_pois_to_sentences(self, poi_gdf: gpd.GeoDataFrame) -> tuple[str, ...]:
#     """Transforms each POI in poi_gdf into the shape: 'At {category} ({section}). {poi_subclass}: {description}'."""
#     # Filter columns to requested map features only
#     map_feature_columns = [col for col in poi_gdf.columns if col in self.map_features]
#     df = poi_gdf[map_feature_columns]
#
#     representations = list()
#     # todo
#     for row in df.itertuples(index=False):
#         non_nan_attributes = (
#             (column, value)
#             for column, value in zip(df.columns, row)
#             if not (isinstance(value, float) and np.isnan(value) and value)
#         )
#
#         poi_representation_parts = list()
#         for i, (category, poi_subclass) in enumerate(non_nan_attributes, start=1):
#             for section in get_sections(
#                 element=poi_subclass,
#                 poi_descriptions=self.poi_descriptions[category],
#             ):
#                 if section is None:
#                     # get description if available
#                     description = self.poi_descriptions[category].get(
#                         poi_subclass, ""
#                     )
#
#                     if poi_subclass == "yes":
#                         representation_part = f"At/In {category}: {description}"
#                     elif category == "office":
#                         representation_part = (
#                             f"At office ({poi_subclass.lower()}): {description}"
#                         )
#                     else:
#                         representation_part = f"At/In {poi_subclass}: {description}"
#                 else:
#                     description = self.poi_descriptions[category][section][
#                         poi_subclass
#                     ]
#                     if poi_subclass == "yes":
#                         representation_part = f"At/In {category}: {description}"
#                     elif category == "shop":
#                         representation_part = f"At/In shop ({section.lower()}). {poi_subclass}: {description}"
#                     elif section == "other":
#                         representation_part = f"At/In {poi_subclass}: {description}"
#                     else:
#                         representation_part = f"At place for {section.lower()}. {poi_subclass}: {description}"
#
#                 poi_representation_parts.append(
#                     representation_part.replace("_", " ")
#                 )
#
#         representations.append(str(" ; ".join(set(poi_representation_parts))))
#
#     return tuple(representations)
#
# def get_sections(element: str, poi_descriptions: dict[str, dict[str, str]] | dict[str, str]) -> Iterator[str]:
#     """Get description section of POI."""
#     returned_element = False
#     if element in poi_descriptions:
#         # descriptions have no sub sections
#         yield None
#
#     for section, values in poi_descriptions.items():
#         if isinstance(values, dict) and element in values:
#             returned_element = True
#             yield section
#
#     if not returned_element:
#         # no section found
#         yield None