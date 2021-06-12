import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
import geopandas as gpd
import requests
import shapely
from shapely.geometry import Point

with st.echo(code_location='below'):
    st.title("Рынок апартаментов Москвы")
    st.header("Я хочу проанализировать Dataset - выгрузку апартаментов с сайта Cian за ноябрь 2020")
    st.subheader('Давайте посмотри что есть в наших данных')
    """
    Используем SQL,чтобы выгрузить только нужные столбцы\n 
    (код для SQL можно посмоттеть в Jupiter Notebook в zip папке)\n
    Price - цена в рублях\n
    House age - возраст дома\n
    dist_to_subway - расстояние до ближайшего метро\n
    lat - широта\n
    lon - долготаn\n
    subway_dist_to_center - расстояние от метро до центра \n
    rooms - кол-во комнат \n
    footage - площадь\n
    AO - административный округ \n
    """
    df1 = pd.read_csv('moscow_apartment_listings.csv')
    df_sub = pd.read_csv('list_of_moscow_metro_stations.csv')
    df1
    del df1['repair']
    del df1['year_built_empty']
    del df1['closest_subway']
    del df1['sg']
    del df1['floor']
    del df1['max_floor']
    del df1['first_floor']
    del df1['last_floor']
    st.header('Используем API для поиска ближайшего метро')
    """
    Как мы видим, в данных нет информации о ближайшей станции метро, поэтому найдем ее сами\n
    Сначала нам нужно спарсить геоданный всех станций метро москвы \n
    Для еэтого воспользуемся API портала data.mos.ru \n
    http://api.data.mos.ru/v1/datasets/1488/rows - используем этот Dataset (код можно посмотреть в Jupiter Notebook)\n
    Теперь у нас есть DataFrame с координатами станций метро \n
    """
    df_sub
    """
    Допустим, что земля в районе москвы плоская \n
    и посчитаем расстояние от апартаметнов до метро как расстояние координатных точек на плоскости \n
    (корень из суммы квадратов разности широты и долготы)\n
    """
    Metro = []
    coor = [list(x) for x in zip(list(df1['lat']), list(df1['lon']))]
    coor_2 = [list(x) for x in zip(list(df_sub['Latitude']), list(df_sub['Longitude']))]
    for i, j in coor:
        a = []
        for x, y in coor_2:
            a.append(math.sqrt((i - x) ** 2 + (j - y) ** 2))
        lat = a.index(min(a))
        Metro.append(df_sub['Name'][lat])
    """
    Теперь добавим в нашу табличку столбец с ближайшим метро - Sub_station\n
    А так же переведем цену в миллионы рублей \n
    Не достает еще одной важной метрики - цена за квадратный метр \n
    Добавим столбец с ценой - price_per_sqm\n
    """
    df1['price'] = df1['price'] / 1000000
    df1['Sub_station'] = Metro
    df1['price_per_sqm'] = df1['price'] / df1['footage'] * 1000
    st.dataframe(df1)
    """
    Теперь посмотрим какие апартаметны есть в нашей базе данных\n
    Для жтого сделаем сводную табличку и посмотрим с помощью функции Pandas.PivotTable\n
    """
    df_totals = pd.DataFrame(
        np.array([[df1['price'].max(), df1['price'].min(), df1['price'].mean(), df1['price'].median()],
                  [df1['price_per_sqm'].max(), df1['price_per_sqm'].min(), df1['price_per_sqm'].mean(),
                   df1['price_per_sqm'].median()],
                  [df1['footage'].max(), df1['footage'].min(), df1['footage'].mean(), df1['footage'].median()]]),
        index=['price, mln rub', 'price_per_sqm, thd rub', 'footage, m2'],
        columns=['max', 'min', 'mean', 'median'])
    st.dataframe(df_totals)

    st.header('Визуализируем данные')
    df_smal = pd.DataFrame(
        [df1['price'], df1['price_per_sqm'], df1['subway_dist_to_center'], df1['dist_to_subway'], df1['footage'],
         df1['house_age']])
    df_small = df_smal.T
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
    axs = axs.flatten()
    index = 0
    for k, v in df_small.items():
        sns.distplot(v, ax=axs[index], color='salmon')
        index += 1
    plt.tight_layout(pad=0.6, w_pad=0.7, h_pad=6.0)
    st.pyplot(fig)
    """Посмотрим как наши апартаменты распределены на карте Москвы\n
    """
    fig = plt.figure(figsize=(32,22))
    sns.jointplot(x=df1.lat.values, y=df1.lon.values, size=10, color='turquoise')
    plt.ylabel('Longitude')
    plt.xlabel('Latitude')
    st.image('Screenshot 2021-06-12 at 17.20.22.png')
    """
    Теперь посмотрим как наши показатели зависят друг от друга \n
    Для этого нарисуем тепловую карту, чтобы посмотреть коэффиценты корреляции между всеми показателями\n
    """
    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(df_small.corr().abs(), annot=True, color='turquoise')
    st.pyplot(fig)
    """
    Нас интересует от чего зависит цена на квадратный метр\n
    Видим значительную положительную зависимость между ценой за квадратный метр и расстоянием до центра\n
    Зависимость между ценой и расстоянием до метро так же положительная, но не такая сильаня\n
    Построим Scatter polts для цены и показателей, которые могу на нее влиять, чтобы посмотреть на зависимость
    """
    st.image('Screenshot 2021-06-12 at 17.36.24.png')
    """
    Код для этих графиков очень перегружает Strimlit,поэтому его можно посмотреть в ноутбуке
    """
    st.header("Агрегируем данные")
    """
    Посмотри теперь на средние значения цены для всех административных оругов
    """

    table1 = pd.pivot_table(df1, values='price_per_sqm', index='AO', aggfunc=np.mean)
    table1
    """
    Как и ожидалось, самым дорогим оказался ЦАО\n
    За ним идем ЗАО\n
    А самым дешовым районом оказался ЮВАО
    """
    """
    Тперь посмотрим на квартиры в ЦАО\n
    Агригируем данные по станции метро
    """
    table2 = pd.pivot_table(df1, values='price_per_sqm', index=['AO', 'Sub_station'], aggfunc=np.mean)
    table2
    """ Самой дорогой в нашем Dataset оказалась станция Чистые Пруды"""
    fig = plt.figure(figsize=(10, 7))
    sns.set(style="white")
    sns.barplot(
        x=table1.index.values,
        y=table1.price_per_sqm.values,
        ci=None,
        color='turquoise')
    st.pyplot(fig)

    table3 = pd.pivot_table(df1[df1['AO'] == 'CAO'], values='price_per_sqm', index='Sub_station', aggfunc=np.mean)
    table3
    fig = plt.figure(figsize=(25, 20))
    sns.set(style="white")
    sns.barplot(
        x=table3.price_per_sqm.values,
        y=table3.index.values,
        ci=None,
        color='turquoise')
    st.pyplot(fig)

    st.header('Используем Geopandas и JSON для построения теплоывой карты цен москвы по районам')
    """
    Streamlit грузит страницу с геопандас больше 10 минут, поэтому код можно посмотреть в ноутбуке, а на сайте я разместила картинку результата \n
    Сначала мы соединили DataFrame карты регионов Москвы и нашу базу данных\n
    Затем посчитали среднее значение цены за квадрайтный метр для каждого района:
    """
    table3=pd.read_csv('table3')
    table3
    """
    Теперь нарисуем это на карте
    """
    st.image('Screenshot 2021-06-12 at 18.57.59.png')

    st.subheader('Работа с Selenium')
    """
    Теперь я хочу проверть зависит ли цена за квадратный метр от расстояния до ближайшей школы\n
    Для этого нам нужно соскреппить геопозицию всех школ Москвы\n
    Используем для этого BeautifulSoup и страницу с названиями и адресами школ Москвы в Википедии\n
    Мы получили все названия и адреса школ Москвы (код внизу или в ноутбуке)
    """
    school_adresses = pd.read_csv('school_adresses')
    school_adresses

    """
    Теперь воспользуемся Гугл картами, чтобы найти их координаты по названию и адресу\n
    Нам понадобится библиотека Selenium\n
    Создадим драйвер, который прогонит серез Гугл карты все адреса и запишет в список ссылки на эти адреса в Гугле\n
    Потом извлечем из ссылок координаты\n
    Получаем таьличку с коорднатами
    """
    new_school_urls = pd.read_csv('school_urls')
    new_school_urls
    """
    Теперь найдем расстояние до ближайшей школы как делали с метро и добавим его в нашу табличку - school_dist
    """
    df_new = pd.read_csv('new_df')
    df_new
    """
    Теперь нарисуем Scatter Plot  и посмотри на зависимость цены и расстояния до школы 
    """
    #FROMhttps://seaborn.pydata.org/examples/regression_marginals.html
    sns.set_theme(style="darkgrid")
    fig = sns.jointplot(x="price_per_sqm", y="school_dist", data=df_new[['price_per_sqm', 'school_dist']],
                      kind="reg", truncate=False,
                      color="turquoise", height=7)
    st.pyplot(fig)
    #ENDFROM
    """
    Теперь мы можем сделать вывод, что цена зависит от расстояния до школы - чем дальше школа, тем ниже цена за квадратный метр
    """




with st.echo(code_location='below'):
    st.subheader('Код для GeoPandas теловой карты')
    #adm_moscow = gpd.read_file("mo.geojson")
    #geometry = [Point(list(x)) for x in zip(list(df1['lon']), list(df1['lat']))]
    #attributes = df1['price_per_sqm']
    #price_map = gpd.GeoDataFrame(attributes, geometry=geometry, crs='EPSG:4326')
    #price_map
    #heatmap = gpd.sjoin(adm_moscow, price_map)
    #table3 = pd.pivot_table(heatmap, values='price_per_sqm', index='NAME', aggfunc=np.mean)
    #table3
    #final_table = adm_moscow.merge(table3, on='NAME')
    #fig = plt.figure()
    #final_table.plot(column='price_per_sqm', figsize=(30, 20), legend=True)
    #st.pyplot(fig)
    st.subheader('Код для Selenium и BeautifulSoup')

    #FROMhttps://medium.com/nuances-of-programming/python-selenium-как-получить-координаты-по-адресам-ea7a78ffdc0d
    #st.subheader('Код для Selenium - геопозиция школ')
    #from bs4 import BeautifulSoup
    #r = requests.get('https://ru.wikipedia.org/wiki/Список_школ_Москвы')
    #page = BeautifulSoup(r.text, 'html.parser')
    #s = page.findAll('li')
    #a = []
    #for i in s:
        #if 'Школа' in i.text:
           # a.append(i.text)
    #b = pd.Series(a)
    #b.to_csv('school_adresses')
    #url_df = ['https://www.google.com/maps/search/' + i for i in b]
    #url_df
    #from selenium import webdriver
    #from tqdm import tqdm_notebook as tqdmn

    #Url_With_Coordinates = []

    #option = webdriver.ChromeOptions()
    #prefs = {'profile.default_content_setting_values': {'images': 2, 'javascript': 2}}
    #option.add_experimental_option('prefs', prefs)

    #driver = webdriver.Chrome("/Users/alinasavelieva/Downloads/chromedriver-3", options=option)

    #for url in tqdmn(url_df, leave=False):
       # driver.get(url)
        #Url_With_Coordinates.append(
         #   driver.find_element_by_css_selector('meta[itemprop=image]').get_attribute('content'))

    #driver.close()
    #new_school_urls = pd.Series(Url_With_Coordinates)[pd.Series(Url_With_Coordinates).str.contains('&zoom=')]
    #lat = [url.split('?center=')[1].split('&zoom=')[0].split('%2C')[0] for url in new_school_urls]
    #lon = [url.split('?center=')[1].split('&zoom=')[0].split('%2C')[1] for url in new_school_urls]
    #ENDFROM
    #school_dist = []
    #coor = [list(x) for x in zip(list(df1['lat']), list(df1['lon']))]
    #coor_schools = [list(x) for x in zip(lat, lon)]
    #for i, j in coor:
        #a = []
        #for x, y in coor_2:
            #a.append(math.sqrt((i - x) ** 2 + (j - y) ** 2))
        #school_dist.append(min(a))
    #df1['school_dist'] = school_dist
    #df1
    #df1.to_csv('new_df')
    st.subheader('Koд для Scatter Plots')
    #FROMhttps://blog.dominodatalab.com/exploring-us-real-estate-values-with-python/
    #from sklearn import preprocessing
    #min_max_scaler = preprocessing.MinMaxScaler()
    #column_sels = ['subway_dist_to_center', 'dist_to_subway', 'footage', 'house_age']
    #x = df_small.loc[:, column_sels]
    #y = df_small['price_per_sqm']
    #x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
    #fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(30, 15))
    #index = 0
    #axs = axs.flatten()
    #for i, k in enumerate(column_sels):
        #sns.regplot(y=y, x=x[k], ax=axs[i], color='salmon')
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    #ENDFROM
