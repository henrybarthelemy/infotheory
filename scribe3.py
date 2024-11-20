from manim import *
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import norm
import math


class BinomialToPoison(Scene):
    def construct(self):
        n = DecimalNumber(10)
        p = ValueTracker(0.2)
        curDel = ValueTracker(2)

        axesP = Axes(
                x_range=[0, 7, 1],
                y_range=[0, 0.6, 0.1], 
                x_length = 9.5,
                axis_config={"color": BLUE,
                "include_numbers": True},
            ).scale(0.6).to_edge(RIGHT, buff = 0.5)

        #creating the graph
        graphP = always_redraw(lambda :
            axesP.plot(lambda x: poisson._pmf(x, curDel.get_value()), color=WHITE)
        )
        labelP = always_redraw(lambda :
            Tex("Y $\sim$ Poisson(" + str(round(curDel.get_value())) + ")").next_to(graphP, DOWN, buff=0.5).scale(0.7)
        )   

        
        chart = always_redraw(lambda :
            BarChart(
                values = self.binomialDis(n.get_value(), p.get_value()),
                bar_names = None,
                y_range = [-0.1, 1, 0.2],
                y_length = 6,
                x_length=  9.5,
                x_axis_config={"font_size": 36},
            ).scale(0.6).to_edge(LEFT, buff = 0.5))
        
        labelB = always_redraw(lambda :
            Tex("X $\sim$ Binomial(" + str(round(n.get_value())) + ", " + str(round(p.get_value(), 3)) + ")").next_to(chart, DOWN, buff=0.5).scale(0.7)
        )   
        


        self.play(Create(axesP))
        self.play(Create(graphP), Create(labelP), Create(labelB), Create(chart))
        self.wait(3) 

    def binomialDis(self, n, p):
        r_values = list(range(n + 1))
        dist = [round(binom.pmf(r, n, p), 2) for r in r_values]
        return dist

    



