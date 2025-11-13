from snappy.database import LinkExteriors
from spherogram import Link
import pandas as pd
import json

import pathlib


def formatPD_Notation(PD_code):
    return str(json.dumps(PD_code).replace(",",";"))

names = []
volumes = []
pd_codes = []
crossing_num = []

for K in LinkExteriors(knots_vs_links = 'links')[:]:
        names.append(K.name())
        pd_codes.append(K.link().PD_code())
        volumes.append(K.volume())
        crossing_num.append(len(K.link().crossings))

data = pd.DataFrame({'Name': names,
              'PD Notation': pd_codes,
              'Volume': volumes,
              'Crossing Number': crossing_num,
              'Determinant': [31] * len(names)})

data['PD Notation'] = data["PD Notation"].apply(formatPD_Notation)

path_to_save = str(pathlib.Path(__file__).parent.resolve()) + "\datasets\link_data.csv"
data.to_csv(path_to_save, index = False)