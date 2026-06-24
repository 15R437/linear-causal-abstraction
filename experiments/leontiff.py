import pymrio

exio_data = pymrio.parse_exiobase3(path="data/IOT_2024_ixi.zip")
exio_data.calc_all()