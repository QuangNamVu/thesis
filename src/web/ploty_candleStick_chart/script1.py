from flask import Flask, render_template

app=Flask(__name__)

@app.route('/')
def home():
    from pandas_datareader import data
    import datetime
    from bokeh.plotting import figure, show, output_file
    from bokeh.embed import components
    from bokeh.resources import CDN

    start = datetime.datetime(2015, 11, 1)
    end = datetime.datetime(2016, 3, 10)

    df = data.DataReader(name="GOOG", data_source="yahoo", start=start, end=end)

    p = figure(x_axis_type='datetime', width=1000, height=300, sizing_mode='scale_both')
    p.title.text = "Candlestick Chart"
    p.grid.grid_line_alpha = 0.4
    hours_12 = 12 * 60 * 60 * 1000

    def inc_dec(c, o):
        if c > o:
            value = "Increase"
        elif c < o:
            value = "Decrease"
        else:
            value = "Equal"
        return value

    df["Status"] = [inc_dec(c, o) for c, o in zip(df.Close, df.Open)]
    df["Middle"] = (df.Open + df.Close) / 2
    # abs always return a positive value
    df["Height"] = abs(df.Open - df.Close)

    # date_increase = df.index[df.Close > df.Open]
    # date_decrease = df.index[df.Close < df.Open]

    # Lines
    p.segment(df.index, df.High, df.index, df.Low, color="black")

    # Rectangles
    p.rect(df.index[df.Status == "Increase"], df.Middle[df.Status == "Increase"], hours_12,
           df.Height[df.Status == "Increase"], fill_color="#CCFFFF", line_color="black")

    p.rect(df.index[df.Status == "Decrease"], df.Middle[df.Status == "Decrease"], hours_12,
           df.Height[df.Status == "Decrease"], fill_color="#FF3333", line_color="black")

    # Script will be index 0 and div1 will be index 1
    script1, div1 = components(p)

    # On our HTML we have to add as well the css and js file of bokeh
    # use index 0 for both of them
    cdn_js = CDN.js_files
    cdn_css = CDN.css_files

    return render_template("home.html", script1=script1, div1=div1, cdn_css=cdn_css[0], cdn_js=cdn_js[0])

@app.route('/about/')
def about():
    return render_template("about.html")


if __name__=="__main__":
    app.run(debug=True)
