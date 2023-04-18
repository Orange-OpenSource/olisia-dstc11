# Software Name : olisia-dstc11
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "license.txt" file for more details.
# Authors: Lucas Druart, Léo Jacqmin

import overpy
import json
import time

def get_names(query):
    names = []
    result = api.query(query)
    for node in result.nodes:
        tags = node.tags
        if 'name' in tags:
            name = tags['name']
            if len(name) > 2 and all(c.isascii() for c in name) and not any(c in name for c in "!@#$%^*()-+?_=,<>/:"):
                names.append(name)
    return names
    
api = overpy.Overpass()
towns = []
hotels = []
restaurants = []

towns = get_names("""
    area[name="United States"];
    nwr[place=town](area);
    out center;
    """)

guest_houses = get_names("""
    area[name="United States"];
    nwr[tourism=guest_house](area);
    out center;
    """)
for town in towns: # Server load too high error when querying the whole US
    try: 
        hotels += get_names(f"""
            area[name="{town}"];
            nwr[tourism=hotel](area);
            out center;
            """)
    except overpy.exception.OverpassTooManyRequests:
        time.sleep(30)
        hotels += get_names(f"""
            area[name="{town}"];
            nwr[tourism=hotel](area);
            out center;
            """)

hotels = hotels + guest_houses

for town in towns: # Server load too high error when querying the whole US
    try:
        restaurants += get_names(f"""
            area[name="{town}"];
            nwr[tourism=restaurant](area);
            out center;
            """)
    except overpy.exception.OverpassTooManyRequests:
        time.sleep(30)
        restaurants += get_names(f"""
            area[name="{town}"];
            nwr[tourism=restaurant](area);
            out center;
            """)

data = {
    'towns': towns,
    'hotels': hotels,
    'restaurants': restaurants    
}

with open('custom_ontology.json', 'w') as f:
    json.dump(data, f)
