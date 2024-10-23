"""
This is a Sketchy, Draft script to learn how to generate pdf reports from python code
"""
import typing
import torch
from fpdf import FPDF
from typing import Dict
from geomloss import SamplesLoss
from matplotlib import pyplot as plt
import numpy as np






def create_report():
    # https://stackoverflow.com/a/51881060/5937273
    # https://stackoverflow.com/questions/51864730/what-is-the-process-to-create-pdf-reports-with-charts-from-a-db
    # https://apitemplate.io/blog/a-guide-to-generate-pdfs-in-python/
    # https://py-pdf.github.io/fpdf2/Tables.html
    # Create Table
    # https://py-pdf.github.io/fpdf2/Tables.html

    TABLE_DATA = (
        ("First name", "Last name", "Age", "City"),
        ("Jules", "Smith", "34", "San Juan"),
        ("Mary", "Ramos", "45", "Orlando"),
        ("Carlson", "Banks", "19", "Los Angeles"),
        ("Lucas", "Cimon", "31", "Saint-Mathurin-sur-Loire"),
    )
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Times", size=16)
    with pdf.table() as table:
        for data_row in TABLE_DATA:
            row = table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.output('table.pdf')
    # pdf = FPDF()
    # pdf.add_page()
    # pdf.set_font('Arial', 'B', 16)
    # pdf.cell(40, 10, 'Hello World!')
    # pdf.output('tuto1.pdf', 'F')
    """
    # https://stackoverflow.com/a/51881060/5937273
    df = pd.DataFrame()
    df['Question'] = ["Q1", "Q2", "Q3", "Q4"]
    df['Charles'] = [3, 4, 5, 3]
    df['Mike'] = [3, 3, 4, 4]

    title("Professor Criss's Ratings by Users")
    xlabel('Question Number')
    ylabel('Score')

    c = [2.0, 4.0, 6.0, 8.0]
    m = [x - 0.5 for x in c]

    xticks(c, df['Question'])

    bar(m, df['Mike'], width=0.5, color="#91eb87", label="Mike")
    bar(c, df['Charles'], width=0.5, color="#eb879c", label="Charles")

    legend()
    axis([0, 10, 0, 8])
    savefig('barchart.png')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('arial', 'B', 12)
    pdf.cell(60)
    pdf.cell(75, 10, "A Tabular and Graphical Report of Professor Criss's Ratings by Users Charles and Mike", 0, 2, 'C')
    pdf.cell(90, 10, " ", 0, 2, 'C')
    pdf.cell(-40)
    pdf.cell(50, 10, 'Question', 1, 0, 'C')
    pdf.cell(40, 10, 'Charles', 1, 0, 'C')
    pdf.cell(40, 10, 'Mike', 1, 2, 'C')
    pdf.cell(-90)
    pdf.set_font('arial', '', 12)
    for i in range(0, len(df)):
        pdf.cell(50, 10, '%s' % (df['Question'].iloc[i]), 1, 0, 'C')
        pdf.cell(40, 10, '%s' % (str(df.Mike.iloc[i])), 1, 0, 'C')
        pdf.cell(40, 10, '%s' % (str(df.Charles.iloc[i])), 1, 2, 'C')
        pdf.cell(-90)
    pdf.cell(90, 10, " ", 0, 2, 'C')
    pdf.cell(-30)
    pdf.image('barchart.png', x=None, y=None, w=0, h=0, type='', link='')
    pdf.output('test.pdf', 'F')
    """


if __name__ == '__main__':
    create_report()
