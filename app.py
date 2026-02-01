import gradio as gr
import pandas as pd
import pickle
import numpy as np


with open('vgsales_model.pkl','rb') as file:
    model=pickle.load(file)

def predict_sales(Name, Platform, Year, Genre, Publisher, NA_Sales,
                EU_Sales, JP_Sales, Other_Sales):
    input_df=pd.DataFrame(
        [[Name, Platform, Year, Genre, Publisher, NA_Sales,
       EU_Sales, JP_Sales, Other_Sales]],
       columns=['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales',
                'EU_Sales', 'JP_Sales', 'Other_Sales']
    )

    prediction=model.predict(input_df)
    return prediction


inputs=[
    gr.Text(label='Name of the Game'),
    gr.Dropdown(['DS','PS2','PS3','Wii','X360','PSP','PS',
                 'PC','XB','GBA','GC','3DS','PSV','PS4','N64',
                 'SNES','XOne','SAT','WiiU','2600','NES','GB','DC','GEN',
                 'NG','SCD','WS','3D0','TG16','GG','PCFX','Others'],label='Platform'),
    gr.Slider(1980,2026,step=1,label='Year'),
    gr.Dropdown(['Action','Sports','Misc','Role-Playing','Shooter','Adventure',''
                'Racing','Platform','Simulation','Fighting','Strategy','Puzzle','Others'],label='Genre'),
    gr.Text(label='Platform'),
    gr.Number(label='NA Sales'),
    gr.Number(label='EU Sales'),
    gr.Number(label='JP Sales'),
    gr.Number(label='Other Sales')
]


app=gr.Interface(
    fn=predict_sales,
    inputs=inputs,
    outputs='text',
    title='Global Price'
)

app.launch(share=True)