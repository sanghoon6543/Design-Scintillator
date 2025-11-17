from tkinter import *
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import UI_Config as UI
import DrawingUtility as UTIL


fd = pathlib.Path(__file__).parent.resolve()
fh = 400
fw = fh*2 + 100
fs = (fw/200, 0.7*fh/100)

class DeviceModelling:
    def __init__(self, window):
        self.window = UI.UI_tkinter.MakeWindow(window, 'Device Modelling', fw, fh, False, background='grey94')
        self.__main__()
        self.InputInfo = self.InputEntryAddress.copy()
        self.DataProcessInfo = self.DataProcessEntryAddress.copy()


    def Event_ApplyInfo(self, frame, entryadress, ftstyle='Calibri', ftsize=24, tickftsize=10):

        ### Update Inputinfo
        for key in entryadress:
            self.InputInfo[key] = UTIL.DataProcessing.GetEntry(entryadress[key])

        ## Make Preview Widget
        if not hasattr(self, 'ax'):
            self.ax, self.canvas = UTIL.FigureConfig.MakeFigureWidget(frame, (0.6*fs[0], 0.95*fs[1]))

        ### Draw Preview Figure
        self.ax.cla()
        UTIL.FigureConfig.FigureConfiguration(self.ax, self.InputInfo, ftstyle, ftsize, tickftsize)
        UTIL.FigureConfig.forceAspect(self.ax, self.InputInfo['xScale'], self.InputInfo['yScale'], aspect=1)
        self.canvas.draw()
        plt.close(plt.gcf())

    def Event_NewFigure(self, inputinfo, ftstyle='Calibri', ftsize=24, tickftsize=14):
        fig, self.drawax = plt.subplots(figsize=fs, tight_layout=True)

        UTIL.FigureConfig.FigureConfiguration(self.drawax, inputinfo, ftstyle, ftsize, tickftsize)
        UTIL.FigureConfig.forceAspect(self.drawax, inputinfo['xScale'], inputinfo['yScale'], aspect=1)
        return self.drawax

    def Event_DrawClipboard(self, ax, color, legendinfo, alphavalue, marker='None', ftsize=16):

        data = UTIL.DataProcessing.ReadClipboard()
        c = next(color)

        ax.plot(data[0], data[1], marker, c=c, alpha=alphavalue)

        ax.legend(legendinfo[:], loc='best', fontsize=ftsize)
        plt.pause(0.001)

    def Event_Calculate(self, ax, inputinfo, dataprocessaddress, ftsize=16):
        ### Update DataProcess Info
        for key in dataprocessaddress:
            self.DataProcessInfo[key] = UTIL.DataProcessing.GetEntry(dataprocessaddress[key])

        SStep, dStep, N, gap, alpha, px, gap2 = float(self.DataProcessInfo['xStep']), float(self.DataProcessInfo['yStep']), \
                                              int(self.DataProcessInfo['N']), float(self.DataProcessInfo['Gap']), \
                                              float(self.DataProcessInfo['abscoeff']), float(self.DataProcessInfo['a']), \
                                              float(self.DataProcessInfo['Gap2'])

        colorstyle = mpl.colormaps[inputinfo['CMapTitleLd_0']]
        vmin, vmax = float(inputinfo['CMap Range_0']), float(inputinfo['CMap Range_1'])

        S = UTIL.DataProcessing.um2cm(np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + SStep, SStep))
        d = UTIL.DataProcessing.um2cm(np.arange(ax.get_ylim()[0], ax.get_ylim()[1] + dStep, dStep))
        g = UTIL.DataProcessing.um2cm(gap)
        g2 = UTIL.DataProcessing.um2cm(gap2)
        px = UTIL.DataProcessing.um2cm(px)

        data = UTIL.DataProcessing.SphericalRadiation_NegativeRefraction(S, d, g, alpha, N, g2)
        # data = UTIL.DataProcessing.SphericalRadiation(S, d, g, alpha, N)

        data_pixelated = UTIL.DataProcessing.pixelation(S, data, px)

        UTIL.DataProcessing.np2clipboard(data=data_pixelated, index=d, columns=S)

        c = ax.imshow(data_pixelated, cmap=colorstyle, alpha=0.8, origin='lower' ,
                      extent = [ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]],
                      vmin = vmin, vmax = vmax)

        ax.cbar = ax.get_figure().colorbar(c, ax=ax)
        ax.cbar.set_label(label=inputinfo['CMapTitleLd_1'], size=ftsize)
        ax.cbar.ax.tick_params(labelsize=ftsize)
        UTIL.FigureConfig.forceAspect(ax, inputinfo['xScale'], inputinfo['yScale'], aspect=1)

        plt.pause(0.001)
        plt.show()

    def Event_CalculateMTF(self, inputinfo, ftsize=16):
        ax = self.Event_NewFigure(inputinfo.copy())

        data = UTIL.DataProcessing.ReadClipboard()
        x = np.array(data.columns).astype(float)
        y = np.array(data.index).astype(float)
        data = UTIL.DataProcessing.df2np(data)
        fft_data = data.copy().astype(complex)
        fft_data[:] = 0

        for k, data_now in enumerate(data):
            fft_x, fft_data[k] = UTIL.DataProcessing.FourierTransform(np.array([x, data_now]))
        fft_data = np.abs(fft_data)
        UTIL.DataProcessing.np2clipboard(data=fft_data, index=y, columns=fft_x)

        colorstyle = mpl.colormaps[inputinfo['CMapTitleLd_0']]

        c = ax.imshow(np.abs(fft_data), cmap=colorstyle, alpha=0.8, origin='lower' ,
                      extent = [min(fft_x), max(fft_x), ax.get_ylim()[0], ax.get_ylim()[1]],
                      vmin = 0, vmax = np.max(fft_data))
        ax.cbar = ax.get_figure().colorbar(c, ax=ax)
        ax.cbar.set_label(label=inputinfo['CMapTitleLd_1'], size=ftsize)
        ax.cbar.ax.tick_params(labelsize=ftsize)
        UTIL.FigureConfig.forceAspect(ax, inputinfo['xScale'], inputinfo['yScale'], aspect=1)

        plt.pause(0.001)
        plt.show()
        asdf = 1

    def __main__(self):
        self.InputInfoFrame = UI.UI_tkinter.MakeFrameLabel(self.window, fw, fh, 0, 0, "Plot Configuration")
        self.OutputFrame = UI.UI_tkinter.MakeFrameLabel(self.window, fw, fh, 1, 0, "Figure Preview")
        self.DataProcessFrame = UI.UI_tkinter.MakeFrameLabel(self.window, fw, fh, 2, 0, "Data Processing")
        ### Input UI

        colspan = 0

        LabelInfos = ["Title", "x-Axis Title", "y-Axis Title", "x Range [\u03BCm]", "Thickness [\u03BCm]", "MajorTick X Y",
                      'CMap Range', "CMap Title Ld"]
        for n, t in enumerate(LabelInfos):
            UI.UI_tkinter.UI_Labels(self.InputInfoFrame, t=t, row=n)

        colspan += 3

        EntryInfos = {'Title': 'Title', 'xAxisTitle': "Distance from Center, s [\u03BCm]", 'yAxisTitle': "Thickness, d [\u03BCm]",
                      'xLim': (0, 1), 'yLim': (0, 1), 'MajorTickXY': (1, 1), 'CMap Range': (0, 1), 'CMapTitleLd': ('RdBu_r', 'Intensity [a.u.]', True)}

        self.InputEntryAddress = {}

        for k, key in enumerate(EntryInfos):
            if type(EntryInfos[key]) is tuple:
                n = EntryInfos[key].__len__()
                for t1, tt in enumerate(EntryInfos[key]):
                    self.InputEntryAddress[key + f'_{t1}'] = UI.UI_tkinter.UI_InputEntry(self.InputInfoFrame, tt, row=k, col=1+t1, width=6)
            else:
                self.InputEntryAddress[key] = UI.UI_tkinter.UI_InputEntry(self.InputInfoFrame, EntryInfos[key], row=k, col=1, colspan=colspan, width=24)

        CBoxInfos = {'xScale': ["Linear", "SymLog"], 'yScale': ["Linear", "SymLog"], 'Grid': ["Grid ON", "Grid Off"]}
        for k, key in enumerate(CBoxInfos):
            self.InputEntryAddress[key] = UI.UI_tkinter.UI_CBox(self.InputInfoFrame, CBoxInfos[key], row=k+3, col=3, width=6, padx=1, pady=1, ftsize=8)

        colspan += 1

        ButtonInfos = ['ApplyInfo']
        self.ButtonAddress = {}
        for n, t in enumerate(ButtonInfos):
            self.ButtonAddress[t] = UI.UI_tkinter.UI_Button(self.InputInfoFrame, t, row=LabelInfos.__len__(), col=0, colspan=colspan, width=30, height=1)

        ### Output UI
        colspan = 0

        self.OutputPlotFrame = UI.UI_tkinter.MakeFrame(self.OutputFrame, 60*fs[0], 95*fs[1], 0, 0, 3)

        colspan += 1
        ButtonInfos = ["New Figure"]
        for n, t in enumerate(ButtonInfos):
            self.ButtonAddress[t] = UI.UI_tkinter.UI_Button(self.OutputFrame, t, row=1, col=0, colspan=colspan, width=20, height=1)

        ### Data Processing UI
        colspan = 0
        LabelInfos = ["x Step", "y Step", 'N', "Gap [\u03BCm]", "\u03B1 [cm\u207b\u00B9]", "Pixel Pitch [\u03BCm]", "Gap 2 [\u03BCm]"]

        colspan += 1
        for n, t in enumerate(LabelInfos):
            UI.UI_tkinter.UI_Labels(self.DataProcessFrame, t=t, row=n)

        EntryInfos = {'xStep': 1, 'yStep': 1, 'N': 1000, 'Gap': 20, 'abscoeff': 476, 'a': 200, 'Gap2':20}

        self.DataProcessEntryAddress = {}

        for k, key in enumerate(EntryInfos):
                self.DataProcessEntryAddress[key] = UI.UI_tkinter.UI_InputEntry(self.DataProcessFrame, EntryInfos[key], row=k, col=1, colspan=colspan, width=10)

        colspan += 1
        ButtonInfos = ["Calculate", "Calculate MTF"]
        for n, t in enumerate(ButtonInfos):
            self.ButtonAddress[t] = UI.UI_tkinter.UI_Button(self.DataProcessFrame, t, row=LabelInfos.__len__() + n, col=0, colspan=colspan, width=24, height=1)

        ### Designate Button Callback Function
        self.ButtonAddress['ApplyInfo'].configure(command=lambda: self.Event_ApplyInfo(self.OutputPlotFrame, self.InputEntryAddress))
        self.ButtonAddress['New Figure'].configure(command=lambda: self.Event_NewFigure(self.InputInfo.copy()))
        self.ButtonAddress['Calculate'].configure(command=lambda: self.Event_Calculate(self.drawax, self.InputInfo.copy(), self.DataProcessEntryAddress))
        self.ButtonAddress['Calculate MTF'].configure(command=lambda: self.Event_CalculateMTF(self.InputInfo.copy()))

if __name__ == '__main__':
    window = Tk()
    DeviceModelling(window)
    window.mainloop()