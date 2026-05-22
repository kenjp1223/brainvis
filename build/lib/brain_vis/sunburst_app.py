import os, math
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import plotly.io as pio
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from .rgba import set_transparency
from typing import Optional, Callable
from .regions import get_subregions  # fallback if user doesn't pass one


def mpl_to_plotly(cmap, n_colors=256):
    return [mcolors.to_hex(cmap(i/n_colors)) for i in range(n_colors)]

def adjust_intensity(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

def convert_to_cmap(image, cmin=-2, cmax=2, colormap=plt.cm.coolwarm):
    norm = Normalize(vmin=cmin, vmax=cmax)
    return colormap(norm(image))[:, :, :3]

def generate_black_contours(id_slice):
    h, w = id_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    unique_ids = np.unique(id_slice)
    for mask_id in unique_ids:
        if mask_id == 0: continue
        binary_mask = np.uint8(id_slice == mask_id) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 0, 255), 1)
    return overlay

def generate_highlight_contours(id_slice, IDs=None):
    if IDs is None:
        IDs = []
    h, w = id_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for mask_id in IDs:
        binary_mask = np.uint8(id_slice == mask_id) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0, 255), 2)
    return overlay

def color_code_image(image_slice, id_slice=None, IDs=None, cmin=-2, cmax=2, colormap=plt.cm.coolwarm):
    if id_slice is None:
        id_slice = image_slice
    base_color = convert_to_cmap(image_slice, cmin=cmin, cmax=cmax, colormap=colormap)
    h, w, _ = base_color.shape
    base_uint8 = (base_color * 255).astype(np.uint8)
    base_rgba = np.dstack([base_uint8, np.full((h, w), 255, dtype=np.uint8)])
    black_overlay = generate_black_contours(id_slice)
    yellow_overlay = generate_highlight_contours(id_slice, IDs)

    final_image = base_rgba.copy()
    mask_black = black_overlay[..., 3] > 0
    final_image[mask_black] = black_overlay[mask_black]
    mask_yellow = yellow_overlay[..., 3] > 0
    final_image[mask_yellow] = yellow_overlay[mask_yellow]

    if id_slice is not None:
        transparency_mask = id_slice == 0
        final_image = set_transparency(final_image, transparency_mask)
    return final_image

def convert_boolean_to_categorical(data_by_region, column='rejected'):
    if data_by_region[column].dtype == bool:
        data_by_region[column] = data_by_region[column].astype(int)
        data_by_region[column] = pd.Categorical(data_by_region[column], categories=[-1, 0, 1])
    elif data_by_region[column].dtype == object:
        data_by_region[column] = data_by_region[column].replace(False, 0).replace(True, 1).replace(np.nan, -1)
        data_by_region[column] = pd.Categorical(data_by_region[column], categories=[-1, 0, 1])
    return data_by_region

def create_dash_app(
    data_by_region,
    data_by_img,
    annotated_atlas_img,
    outputpath,
    get_subregions_func: Optional[Callable] = None,
    colormap=plt.cm.coolwarm,
    tree_node_names='acronym',
    tree_node_parents='parent_acronym',
    data_variable='rejected',
    sunburst_continuous_scale=None,
    sunburst_range_color=None,
    uniformtext_minsize=10,
    uniformtext_mode='hide',
    data_cmin=None,
    data_cmax=None,
    **kwargs
    ):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    # Normalization range for data_by_img display (vmin/vmax for colormap)
    if data_cmin is None:
        data_cmin = -7.5
    if data_cmax is None:
        data_cmax = 7.5
    if sunburst_continuous_scale is None:
        sunburst_continuous_scale = mpl_to_plotly(plt.cm.viridis)
    data_by_region = convert_boolean_to_categorical(data_by_region, column=data_variable)
    if get_subregions_func is None:
        # Use the default implementation from regions.py
        get_subregions_func = lambda df, rid, return_original=True: get_subregions(
            df, rid, return_original=return_original
        )

    if data_by_region[data_variable].dtype != 'category':
        fig_sunburst = px.sunburst(
            data_by_region,
            names=tree_node_names,
            parents=tree_node_parents,
            color=data_variable,
            color_continuous_scale=sunburst_continuous_scale,
            range_color=sunburst_range_color,
            hover_data={tree_node_names: True, 'name': True},
        )
        fig_sunburst.update_layout(
            coloraxis_colorbar=dict(thickness=10, len=0.5),
            uniformtext=dict(minsize=uniformtext_minsize, mode=uniformtext_mode)
        )
    else:
        fig_sunburst = px.sunburst(
            data_by_region,
            names=tree_node_names,
            parents=tree_node_parents,
            color=data_variable,
            color_discrete_map={-1: 'white', 1: 'magenta', 0: 'grey'},
            hover_data={tree_node_names: True, 'name': True},
        )
        fig_sunburst.update_layout(uniformtext=dict(minsize=uniformtext_minsize, mode=uniformtext_mode))

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='sunburst-plot', figure=fig_sunburst, style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                html.Div(children=[dcc.Graph(id='image-plot', style={'width': '100%'})],
                         style={'position': 'relative', 'width': '100%'}),
                html.Div(children=[
                        html.Button("Z -10", id="z-minus-10", n_clicks=0),
                        html.Button("Z -1", id="z-minus-1", n_clicks=0),
                        html.Button("Z +1", id="z-plus-1", n_clicks=0),
                        html.Button("Z +10", id="z-plus-10", n_clicks=0)
                    ],
                    style={'position': 'absolute','top': '50px','left': '150px','backgroundColor': 'rgba(0,0,0,0.5)',
                           'padding': '5px','borderRadius': '5px'}),
                html.Div(children=[
                        html.Button("Save as PNG", id="export-png-button", n_clicks=0),
                        html.Button("Save as PDF", id="export-pdf-button", n_clicks=0),
                        html.Button("Export as HTML", id="export-html-button", n_clicks=0)
                    ],
                    style={'position': 'absolute','bottom': '20px','right': '90px','backgroundColor': 'rgba(0,0,0,0.5)',
                           'padding': '5px','borderRadius': '5px'})
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'position': 'relative'})
        ]),
        html.Div(id='save-status', style={'textAlign': 'center','fontSize': '14px','color': 'green','padding': '5px','marginTop': '5px'}),
        dcc.Store(id='region_info_store')
    ])

    @app.callback(
        [Output('image-plot', 'figure'), Output('region_info_store', 'data')],
        [Input('sunburst-plot', 'clickData'), Input('z-minus-10', 'n_clicks'),
         Input('z-minus-1', 'n_clicks'), Input('z-plus-1', 'n_clicks'), Input('z-plus-10', 'n_clicks')],
        State('region_info_store', 'data')
    )
    def update_image(clickData, n_zm10, n_zm1, n_zp1, n_zp10, region_info):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        if triggered_id == 'sunburst-plot' and clickData is not None:
            selected_acronym = clickData['points'][0].get('label')
            selected_row = data_by_region.loc[data_by_region[tree_node_names] == selected_acronym]
            if selected_row.empty:
                return go.Figure(), None
            selected_id = selected_row['id'].values[0]
            selected_name = selected_row['name'].values[0]

            tatlas_df = get_subregions_func(data_by_region, selected_id, return_original=True)
            IDs = tatlas_df['id'].values
            locations = np.where(np.isin(annotated_atlas_img, IDs))
            z_index = int(np.bincount(locations[0]).argmax()) if locations[0].size > 0 else 0
            region_info = {'selected_acronym': selected_acronym, 'selected_id': selected_id, 'selected_name': selected_name,
                           'IDs': IDs.tolist(), 'current_z_index': z_index}
        elif triggered_id in ['z-minus-10','z-minus-1','z-plus-1','z-plus-10'] and region_info is not None:
            delta = {'z-minus-10': -10, 'z-minus-1': -1, 'z-plus-1': 1, 'z-plus-10': 10}.get(triggered_id, 0)
            region_info['current_z_index'] = int(region_info.get('current_z_index', 0)) + delta

        if region_info is None:
            return go.Figure(), None

        z_index = region_info.get('current_z_index', 0)
        max_z = data_by_img.shape[0] - 1
        z_index = max(0, min(z_index, max_z))
        region_info['current_z_index'] = z_index

        data_slice = data_by_img[z_index, :, :]
        image_slice = annotated_atlas_img[z_index, :, :]

        rgb_image_slice = color_code_image(data_slice, image_slice, IDs=np.array(region_info['IDs']), colormap=colormap, cmin=data_cmin, cmax=data_cmax)
        region_info['rgb_image_slice'] = rgb_image_slice
        fig_image = px.imshow(rgb_image_slice)

        id_to_name = {row['id']: row['name'] for _, row in data_by_region.iterrows()}
        name_data = np.vectorize(id_to_name.get)(image_slice)
        name_data[name_data == None] = 'background'
        customdata = np.stack([data_slice, name_data], axis=-1)
        fig_image.update_traces(
            customdata=customdata,
            hovertemplate=(
                "X: %{x} <br>Y: %{y} <br>Value: %{customdata[0]:.3f} <br>Region: %{customdata[1]}<extra></extra>"
            )
        )
        fig_image.update_layout(
            title=f"{region_info['selected_name']}, Z-Slice: {z_index}",
            coloraxis_showscale=False,
            hovermode='x unified',
            hoverlabel=dict(bgcolor="rgba(255,255,255,0.75)", font_color="grey", font_size=12, namelength=0)
        )
        return fig_image, region_info

    @app.callback(
        Output("save-status", "children"),
        [Input("export-png-button", "n_clicks"), Input("export-pdf-button", "n_clicks"), Input("export-html-button", "n_clicks")],
        [State("sunburst-plot", "figure"), State("image-plot", "figure"), State("region_info_store", "data")]
    )
    def export_figures(png_clicks, pdf_clicks, html_clicks, sunburst_fig_data, image_fig_data, region_info):
        ctx = dash.callback_context
        if not ctx.triggered or sunburst_fig_data is None or image_fig_data is None or region_info is None:
            raise dash.exceptions.PreventUpdate
        selected_acronym = region_info.get('selected_acronym', "unknown")
        rgb_image_slice = region_info.get('rgb_image_slice')
        if rgb_image_slice is None:
            return "Error: RGB image slice not found in region info."
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        high_res_scale = 8
        sunburst_fig = go.Figure(sunburst_fig_data)
        image_fig = go.Figure(image_fig_data)
        if button_id == "export-png-button":
            sunburst_filename = os.path.join(outputpath, f"sunburst_{selected_acronym}.png")
            image_filename = os.path.join(outputpath, f"image_{selected_acronym}.tif")
            try:
                sunburst_fig.write_image(sunburst_filename, format="png", scale=high_res_scale)
                from tifffile import imwrite
                imwrite(image_filename, np.array(rgb_image_slice, dtype=np.uint8))
                return f"PNG file saved: {sunburst_filename}, TIFF file saved: {image_filename}"
            except Exception as e:
                return f"Error saving files: {e}"
        elif button_id == "export-pdf-button":
            sunburst_filename = os.path.join(outputpath, f"sunburst_{selected_acronym}.pdf")
            image_filename = os.path.join(outputpath, f"image_{selected_acronym}.tif")
            try:
                sunburst_fig.write_image(sunburst_filename, format="pdf", scale=high_res_scale)
                from tifffile import imwrite
                imwrite(image_filename, np.array(rgb_image_slice, dtype=np.uint8))
                return f"PDF file saved: {sunburst_filename}, TIFF file saved: {image_filename}"
            except Exception as e:
                return f"Error saving files: {e}"
        elif button_id == "export-html-button":
            sunburst_html = pio.to_html(sunburst_fig, full_html=True, include_plotlyjs='cdn')
            image_html = pio.to_html(image_fig, full_html=True, include_plotlyjs='cdn')
            sunburst_filename = os.path.join(outputpath, f"sunburst_{selected_acronym}.html")
            image_filename = os.path.join(outputpath, f"image_{selected_acronym}.html")
            try:
                with open(sunburst_filename, "w", encoding="utf-8") as f:
                    f.write(sunburst_html)
                with open(image_filename, "w", encoding="utf-8") as f:
                    f.write(image_html)
                return f"HTML files saved: {sunburst_filename} and {image_filename}"
            except Exception as e:
                return f"Error saving HTML files: {e}"
    return app

def run_app(*args, **kwargs):
    app = create_dash_app(*args, **kwargs)
    return app
