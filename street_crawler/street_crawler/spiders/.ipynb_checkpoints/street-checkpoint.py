from scrapy.shell import inspect_response
from urllib.parse import urlencode
from unidecode import unidecode
from tqdm import tqdm
from PIL import Image

import geopandas as gpd
import numpy as np
import requests
import scrapy
import random
import json
import pdb
import os

from ..items import Street
from ..settings import IMAGES_STORE


cidades = {
    "sp": [
        "Tapiraí",
        "Sete Barras",
        "São Lourenço da Serra",
        "Ribeira",
        "Registro",
        "Pedro de Toledo",
        "Pariquera-Açu",
        "Miracatu",
        "Juquitiba",
        "Juquiá",
        "Jacupiranga",
        "Itariri",
        "Itapirapuã Paulista",
        "Itaoca",
        "Iporanga",
        "Ilha Comprida",
        "Iguape",
        "Eldorado",
        "Cananéia",
        "Cajati",
        "Barra do Turvo",
        "Barra do Chapéu",
        "Apiaí",
    ],
    "pr": [
        "Tunas do Paraná",
        "Rio Branco do Sul",
        "Itaperuçu",
        "Doutor Ulysses",
        "Cerro Azul",
        "Bocaiúva do Sul",
        "Adrianópolis",
    ],
}


class StreetSpider(scrapy.Spider):
    name = "street"
    allowed_domains = [
        "cep.guiamais.com.br",
        "maps.googleapis.com",
        "geo0.ggpht.com",
        "geo1.ggpht.com",
        "geo2.ggpht.com",
        "geo3.ggpht.com",
    ]

    def start_requests(self):
        # Search for streets in city or for lat long coordinates
        SEARCH_STREETS = False
        myfunc = self.search_street() if SEARCH_STREETS else self.search_lat_long()
        return myfunc

    def search_lat_long(self):
        STREET_API_URL = "https://maps.googleapis.com/maps/api/streetview"
        sectors = gpd.read_file("../demo/vale_ribeira.shp")
        for _, row in tqdm(
            sectors.iterrows(),
            total=sectors.shape[0],
            leave=False,
            desc="Searching sectors",
            unit="sectors",
        ):
            city = row["NM_MUNICIP"]
            uf = "unknown"
            minx, miny, maxx, maxy = row["geometry"].bounds
            for x in tqdm(np.linspace(minx, maxx, 20), desc="X", leave=False):
                for y in tqdm(np.linspace(miny, maxy, 20), desc="Y", leave=False):
                    params = {
                        "location": f"{y:.4f}, {x:.4f}",
                        "key": os.environ["STREET_VIEW_STATIC_API_KEY"],
                    }
                    yield scrapy.Request(
                        url=f"{STREET_API_URL}/metadata?{urlencode(params)}",
                        callback=self.parse_street,
                        cb_kwargs={"street_name": "unknown", "city": city, "uf": uf},
                    )

    def search_street(self):
        self.count_streets = {"sp": {}, "pr": {}}
        base_url = "http://cep.guiamais.com.br/busca/{}-{}"
        for uf, cidade_lista in cidades.items():
            for cidade in cidade_lista:
                self.count_streets[uf][cidade] = {"total": 0, "imagens": 0}
                cidade_format = unidecode(cidade.lower()).replace(" ", "-")
                yield scrapy.Request(
                    url=base_url.format(cidade_format, uf),
                    callback=self.parse_city,
                    cb_kwargs={"uf": uf, "city": cidade},
                )

    def parse_city(self, response, uf, city):
        STREET_API_URL = "https://maps.googleapis.com/maps/api/streetview"
        for street in response.selector.xpath("//table/tbody/tr"):
            street_name = street.xpath("td[1]/a/text()").get()
            if not street_name.strip():
                continue
            params = {
                "location": f'"{street_name}, {city}" - {uf}, Brasil',
                "key": os.environ["STREET_VIEW_STATIC_API_KEY"],
            }

            self.count_streets[uf][city]["total"] += 1
            # metadata = requests.get(f"{STREET_API_URL}/metadata?{urlencode(params)}")
            # if metadata.status_code == 200:
            #     meta_report = json.loads(metadata.text)
            #     if meta_report.get("status", "") == "OK":
            #         self.count_streets[uf][city]["imagens"] += 1

            yield scrapy.Request(
                url=f"{STREET_API_URL}/metadata?{urlencode(params)}",
                callback=self.parse_street,
                cb_kwargs={"street_name": street_name, "city": city, "uf": uf},
            )
        with open("count_streets.json", "w") as f:
            json.dump(self.count_streets, f)

        next_link = response.selector.xpath("//ul[contains(@class, 'pager')]/li/a")[-1]
        next_link_name = next_link.xpath("span/text()").get().strip()
        if next_link_name == "próximo":
            yield response.follow(
                next_link.xpath("@href").get(),
                callback=self.parse_city,
                cb_kwargs={"uf": uf, "city": city},
            )

    def parse_street(self, response, uf, city, street_name):
        data = json.loads(response.text)
        if data["status"] == "OK":
            for direction in [0, 90, 180, 270]:
                filename = f"{data['pano_id']}-{direction}.jpg"
                street = Street()
                street["uf"] = uf
                street["city"] = city
                street["name"] = street_name
                street["latitude"] = data["location"]["lat"]
                street["longitude"] = data["location"]["lng"]
                street["pano_id"] = data["pano_id"]
                street["direction"] = direction
                for y in [0, 1]:
                    params = {
                        "cb_client": "maps_sv.tactile",
                        "authuser": 0,
                        "hl": "en",
                        "gl": "br",
                        "x": direction // 90,
                        "y": y,
                        "zoom": 2,
                        "nbt": None,
                        "fover": 0,
                        "output": "tile",
                        "panoid": data["pano_id"],
                    }
                    yield scrapy.Request(
                        url=f"https://geo{random.randint(0, 3)}.ggpht.com/cbk?{urlencode(params)}",
                        callback=self.parse_image,
                        cb_kwargs={"filename": f"{filename}-{y}"},
                    )
                street["filename"] = filename
                yield street

    def parse_image(self, response, filename):
        unifile = os.path.join(IMAGES_STORE, filename)
        with open(unifile, "wb") as f:
            f.write(response.body)
        unifile = unifile[:-2]
        if os.path.exists(unifile + "-0") and os.path.exists(unifile + "-1"):
            img0 = np.array(Image.open(unifile + "-0"))
            img1 = np.array(Image.open(unifile + "-1"))
            Image.fromarray(np.concatenate((img0, img1))[256 : 256 + 512, :, :]).save(
                unifile
            )
            os.remove(unifile + "-0")
            os.remove(unifile + "-1")

    def closed(self, reason):
        print(self.count_streets)
