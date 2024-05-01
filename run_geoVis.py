import pandas as pd
import geopandas as gpd
import plotly.express as px

from utils.eval import cv_mean_absolute_error_wAbs


def tractLevelMap(geo_df, colorCol, saveAddr, color_discrete_map, token):
    fig = px.choropleth_mapbox(geo_df,
                               geojson=geo_df.geometry,
                               locations=geo_df.index,
                               color=colorCol,

                               color_discrete_map=color_discrete_map,
                               category_orders={colorCol: list(color_discrete_map.keys())[::-1]},
                               # color_continuous_scale='Blues',

                               center={"lat": 34.165301, "lon": -118.219155},
                               opacity=0.8,
                               )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox={
                          'accesstoken': token,
                          'style': "light",
                      },
                      )
    fig.write_html(saveAddr)
    return


def vis_climateCoor(climateID, dir_corrMatrix):
    # get climate zone geo
    climateZone = gpd.read_file('./data/geo_data/coarse_grid.geojson')
    climateZone['id.grid.coarse'] = climateZone['id.grid.coarse'].astype(int)
    climateZone_geo = climateZone[['id.grid.coarse', 'geometry']]
    # get coor for selected climate zone
    climateCorr = pd.read_csv(dir_corrMatrix).set_index('Climate')
    climateCorrSelect = climateCorr[climateCorr.index == climateID].T.reset_index().rename({'index': 'climate',
                                                                                            climateID: 'corr'}, axis=1)
    climateCorrSelect.climate = climateCorrSelect.climate.astype(int)
    # merge
    climate_merged = climateZone_geo.merge(climateCorrSelect, how='right', right_on='climate', left_on='id.grid.coarse')
    # vis
    climate_merged.plot(column='corr')


if __name__ == '__main__':

    # get mapbox token
    with open('./utils/mapbox_token.txt', 'r') as file:
        mapbox_token = file.readline().strip()
    # exp name
    exp_name = 'energyElec_biLSTM_10PerData_2023-08-15-16-09-06'
    # analysis name
    analysis = 'tractLevel_building_stock'

    if analysis == 'tractLevel_metric':
        # get data on tracts level
        tracts_estimate = pd.read_csv(
            './saved/estimates_tracts/' + exp_name + '/tractsDF.csv').drop('Unnamed: 0', axis=1)
        tracts_estimate.geoid = tracts_estimate.geoid.astype(str)
        # get tracts geometry
        tracts_geo = gpd.read_file('./data/geo_data/tract.geojson')
        tracts_geo['id.tract'] = tracts_geo['id.tract'].str[1:]
        # combine metric and geo
        tractMetric = tracts_estimate.groupby('geoid').apply(
            lambda g: cv_mean_absolute_error_wAbs(g.true, g.estimate)).to_frame('nMAE').reset_index()
        tractMetricGeo = tractMetric.merge(tracts_geo[['id.tract', 'geometry']], how='left', left_on='geoid',
                                           right_on='id.tract').set_index('geoid')
        tractMetricGeo = gpd.GeoDataFrame(tractMetricGeo, crs="EPSG:4326", geometry=tractMetricGeo.geometry)

        # norm
        # tractMetricGeo['Normalized nMAE'] = tractMetricGeo['nMAE'] / tractMetricGeo['nMAE'].nlargest(2).iloc[-1]
        tractMetricGeo['nMAE Quantiles'] = pd.qcut(tractMetricGeo['nMAE'],
                                                    q=[0, .2, .4, .6, .8, 1],
                                                    labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1'])
        # draw
        color_discrete_map = {
            '0-0.2': '#eff3ff',
            '0.2-0.4': '#bdd7e7',
            '0.4-0.6': '#6baed6',
            '0.6-0.8': '#3182bd',
            '0.8-1': '#08519c',
        }
        tractLevelMap(tractMetricGeo,
                      'nMAE Quantiles',
                      './paper/figs/map_tractCVMAE_quantiles.html',
                      color_discrete_map,
                      mapbox_token)

    if analysis == 'tractLevel_building_stock':
        # calculate residential building area percentage
        building_meta = pd.read_csv(
            './data/building_metadata/building_metadata.csv')

        tract_building_area = building_meta[['id.tract', 'building.area.m2']].groupby(['id.tract']).sum()
        tract_building_type_area = building_meta[['id.tract', 'building.type', 'building.area.m2']].groupby(
            ['id.tract', 'building.type']).sum().reset_index()

        tract_resi_per = tract_building_type_area.merge(tract_building_area, on='id.tract', how='left')
        tract_resi_per['percentage'] = tract_resi_per['building.area.m2_x'] / tract_resi_per['building.area.m2_y']
        tract_resi_per = tract_resi_per[tract_resi_per['building.type'].isin(['single-family', 'multi-family'])]
        tract_resi_per = tract_resi_per[['id.tract', 'percentage']].groupby('id.tract').sum()

        tract_id = building_meta[['id.tract']].groupby('id.tract').count().reset_index()
        tract_resi_per = tract_resi_per.merge(tract_id, on='id.tract', how='right')
        tract_resi_per = tract_resi_per.fillna(0)
        tract_resi_per['id.tract'] = tract_resi_per['id.tract'].astype(str)

        # get data on tracts level and filter tract_resi_per_geo
        tracts_estimate = pd.read_csv(
            './saved/estimates_tracts/' + exp_name + '/tractsDF.csv').drop('Unnamed: 0', axis=1)
        tracts_estimate.geoid = tracts_estimate.geoid.astype(str)
        tracts_4_test = tracts_estimate.geoid.unique().tolist()

        tract_resi_per = tract_resi_per[tract_resi_per['id.tract'].isin(tracts_4_test)]

        # get tracts geometry
        tracts_geo = gpd.read_file('./data/geo_data/tract.geojson')
        tracts_geo['id.tract'] = tracts_geo['id.tract'].str[1:]

        # combine metric and geo
        tract_resi_per_geo = tract_resi_per.merge(tracts_geo[['id.tract', 'geometry']], how='left',
                                                  on='id.tract').set_index('id.tract')
        tract_resi_per_geo = gpd.GeoDataFrame(tract_resi_per_geo, crs="EPSG:4326", geometry=tract_resi_per_geo.geometry)

        # norm
        tract_resi_per_geo['Percentage Quantiles'] = pd.qcut(tract_resi_per_geo['percentage'],
                                                              q=[0, .2, .4, .6, .8, 1],
                                                              labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8',
                                                                      '0.8-1'])

        # draw
        color_discrete_map = {
            '0-0.2': '#eff3ff',
            '0.2-0.4': '#bdd7e7',
            '0.4-0.6': '#6baed6',
            '0.6-0.8': '#3182bd',
            '0.8-1': '#08519c',
        }
        tractLevelMap(tract_resi_per_geo,
                      'Percentage Quantiles',
                      './paper/figs/map_tract_resi_per_quantiles.html',
                      color_discrete_map,
                      mapbox_token)

    if analysis == 'tractLevel_metric_stock_correlation':
        # calculate residential building area percentage
        building_meta = pd.read_csv(
            './data/building_metadata/building_metadata.csv')

        tract_building_area = building_meta[['id.tract', 'building.area.m2']].groupby(['id.tract']).sum()
        tract_building_type_area = building_meta[['id.tract', 'building.type', 'building.area.m2']].groupby(
            ['id.tract', 'building.type']).sum().reset_index()

        tract_resi_per = tract_building_type_area.merge(tract_building_area, on='id.tract', how='left')
        tract_resi_per['percentage'] = tract_resi_per['building.area.m2_x'] / tract_resi_per['building.area.m2_y']
        tract_resi_per = tract_resi_per[tract_resi_per['building.type'].isin(['single-family', 'multi-family'])]
        tract_resi_per = tract_resi_per[['id.tract', 'percentage']].groupby('id.tract').sum()

        tract_id = building_meta[['id.tract']].groupby('id.tract').count().reset_index()
        tract_resi_per = tract_resi_per.merge(tract_id, on='id.tract', how='right')
        tract_resi_per = tract_resi_per.fillna(0)
        tract_resi_per['id.tract'] = tract_resi_per['id.tract'].astype(str)

        # get data on tracts level and filter tract_resi_per_geo
        tracts_estimate = pd.read_csv(
            './saved/estimates_tracts/' + exp_name + '/tractsDF.csv').drop('Unnamed: 0', axis=1)
        tracts_estimate.geoid = tracts_estimate.geoid.astype(str)
        tracts_4_test = tracts_estimate.geoid.unique().tolist()

        tract_resi_per = tract_resi_per[tract_resi_per['id.tract'].isin(tracts_4_test)]

        # get tract metric
        tract_metric = tracts_estimate.groupby('geoid').apply(
            lambda g: cv_mean_absolute_error_wAbs(g.true, g.estimate)).to_frame('nMAE').reset_index()

        # combine and plot
        tracts_metric_res_per = tract_resi_per.merge(tract_metric, left_on='id.tract', right_on='geoid', how='left')
        fig = px.scatter(tracts_metric_res_per, x='percentage', y='nMAE')
        fig.update_layout(template='seaborn', xaxis_title='Percentage of Residential Building Area')
        fig.write_html('./paper/figs/scatter_nMAE_building_stock.html')
        print()

    if analysis == 'climateZone_corr':
        vis_climateCoor(pd.read_csv('./saved/climateCorr/climateCorr.csv').Climate.unique().tolist()[15],
                        './saved/climateCorr/climateCorr.csv')

    if analysis == 'drawGrid':
        grid_geo = gpd.read_file('./data/geo_data/coarse_grid.geojson')
        grid_geo['color'] = 1

        fig = px.choropleth_mapbox(grid_geo,
                                   geojson=grid_geo.geometry,
                                   color="color",
                                   locations=grid_geo.index,
                                   color_continuous_scale="Viridis",
                                   center={"lat": 33.994512, "lon": -118.228486},
                                   opacity=0.3,
                                   zoom=7.8,
                                   )
        fig.add_scattermapbox(
            lat=grid_geo.geometry.centroid.y.tolist(),
            lon=grid_geo.geometry.centroid.x.tolist(),
            mode='markers+text',
            text=grid_geo['id.grid.coarse'].astype(int).astype(str),
            marker_size=0.5,
            textfont=dict(size=12, color='#3F3F3F'),
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          mapbox={
                              'accesstoken': mapbox_token,
                              'style': "light",
                          },
                          showlegend=False,
                          )
        fig.data = (fig.data[1], fig.data[0])
        fig.write_html('./paper/figs/mapGrids.html')

    if analysis == 'drawTract':
        # draw a selected census tract on map
        tract_geo = gpd.read_file('./data/geo_data/tract.geojson')
        tract_geo_select = tract_geo[tract_geo['id.tract'].astype(float) == 6037621201]
        tract_geo_select['color'] = [1] * len(tract_geo_select)

        fig = px.choropleth_mapbox(tract_geo_select,
                                   geojson=tract_geo_select.geometry,
                                   color="color",
                                   locations=tract_geo_select.index,
                                   color_continuous_scale="Viridis",
                                   center={"lat": 33.994512, "lon": -118.228486},
                                   opacity=0.3,
                                   zoom=7.8,
                                   )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                          mapbox={
                              'accesstoken': mapbox_token,
                              'style': "light",
                          },
                          showlegend=False,
                          )
        fig.write_html('./paper/figs/tract6037621201.html')
