import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, Locator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from tkinter import *




class FigureConfig:
    @staticmethod
    def MakeFigureWidget(frame, figuresize):
        fig, ax = plt.subplots(figsize=figuresize, tight_layout=True)
        tk_plt = FigureCanvasTkAgg(fig, frame)
        tk_plt.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=1)
        plt.close(fig)
        return ax, tk_plt

    @staticmethod
    def MakeFigure(figuresize, legendinfo, colorstyle):
        colorstyle = mpl.colormaps[colorstyle]
        color = iter(colorstyle(np.linspace(1, 0, legendinfo.__len__())))
        fig, ax = plt.subplots(figsize=figuresize, tight_layout=True)
        return ax, color

    @staticmethod
    def FigureConfiguration(ax, inputinfo, ftstyle='Calibri', ftsize=24, tickftsize=10):
        ax.set_title(inputinfo['Title'], font=ftstyle, fontsize=ftsize)
        ax.set_xlabel(inputinfo['xAxisTitle'], font=ftstyle, fontsize=ftsize)
        ax.set_ylabel(inputinfo['yAxisTitle'], font=ftstyle, fontsize=ftsize)
        ax.set_xlim((float(inputinfo['xLim_0']), float(inputinfo['xLim_1'])))
        ax.set_ylim((float(inputinfo['yLim_0']), float(inputinfo['yLim_1'])))

        if inputinfo['xScale'] == 'Linear':
            ax.xaxis.set_major_locator(MultipleLocator(float(inputinfo['MajorTickXY_0'])))
        elif inputinfo['xScale'] == 'SymLog':
            FigureConfig.SymlogScale(ax, float(inputinfo['xLim_0']), float(inputinfo['xLim_1']), float(inputinfo['xLim_0']), float(inputinfo['MajorTickXY_0']), 'x')

        if inputinfo['yScale'] == 'Linear':
            ax.yaxis.set_major_locator(MultipleLocator(float(inputinfo['MajorTickXY_1'])))
        elif inputinfo['yScale'] == 'SymLog':
            FigureConfig.SymlogScale(ax, float(inputinfo['yLim_0']), float(inputinfo['yLim_1']), float(inputinfo['yLim_0']), float(inputinfo['MajorTickXY_1']), 'y')

        if inputinfo['Grid'] == 'Grid ON':
            ax.grid(True)
        ax.tick_params(axis='x', labelsize=tickftsize)
        ax.tick_params(axis='y', labelsize=tickftsize)
        plt.tight_layout()

    @staticmethod
    def SymlogScale(ax, linthresh, up, dn, interval, axis):

        tickorder = np.arange(np.log10(up), np.log10(dn), -np.log10(interval))
        if tickorder[-1] != np.log10(dn):
            tickorder = np.append(tickorder, np.log10(dn))
        tick = 10 ** tickorder

        if axis == 'x':
            ax.set_xscale("symlog", linthresh=linthresh)
            ax.set_xticks(tick)
            ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh, (dn, up)))

        if axis == 'y':
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.set_yticks(tick)
            ax.yaxis.set_minor_locator(MinorSymLogLocator(linthresh, (dn, up)))


    @staticmethod
    def forceAspect(ax, xscale, yscale, aspect=1):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if xscale == yscale:
            ax.set_aspect(abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0])) / aspect)
        elif xscale == "SymLog" and yscale == "Linear":
            ax.set_aspect(0.1*abs((np.log10(xlim[1]) - np.log10(xlim[0])) / (ylim[1] - ylim[0])) / aspect)
        elif xscale == "Linear" and yscale == "SymLog":
            ax.set_aspect(10*abs((xlim[1] - xlim[0]) / (np.log10(ylim[1]) - np.log10(ylim[0]))) / aspect)


class DataProcessing:
    @staticmethod
    def ReadClipboard():
        return pd.read_clipboard()

    @staticmethod
    def df2np(df):
        return np.array(df)

    @staticmethod
    def np2clipboard(data, index=None, columns=None):
        df = pd.DataFrame(data=data, index=index, columns=columns)
        df.to_clipboard(excel=True)

    @staticmethod
    def GetEntry(EntryAddress):
        return EntryAddress.get()

    @staticmethod
    def um2cm(value):
        return value*1E-4

    @staticmethod
    def cm2um(value):
        return value*1E4

    @staticmethod
    def pixelation(S, f, px):

        f_pixelated = f.copy()
        f_pixelated[:] = 0
        N_Neg = DataProcessing.__findN(S.min(), px)
        N_Pos = DataProcessing.__findN(S.max(), px)
        N_Neg_End = DataProcessing.__findNend(S.min(), px)
        N_Pos_End = DataProcessing.__findNend(S.max(), px)

        nPx = int(px / np.round(np.abs(S[0] - S[1]), 10))

        f_left, data = DataProcessing.__Truncation(f, nPx, N_Neg, N_Neg_End, 'left')
        f_right, data = DataProcessing.__Truncation(data, nPx, N_Pos, N_Pos_End, 'right')

        if data.shape[-1] % 2 == 1:
            data = np.delete(data, data.shape[-1]//2, axis=1)
            data = DataProcessing.__Pixelation(data, nPx)
            data = np.insert(data, data.shape[1]//2, f[:, f.shape[1]//2], axis=1)
            data = np.c_[np.repeat(np.average(f_left, axis=-1)[:, np.newaxis], f_left.shape[-1], axis=-1), data]
            data = np.c_[data, np.repeat(np.average(f_right, axis=-1)[:, np.newaxis], f_right.shape[-1], axis=-1)]
            return data

        data = DataProcessing.__Pixelation(data, nPx)
        data = np.c_[np.repeat(np.average(f_left, axis=-1)[:, np.newaxis], f_left.shape[-1], axis=-1), data]
        data = np.c_[data, np.repeat(np.average(f_right, axis=-1)[:, np.newaxis], f_right.shape[-1], axis=-1)]
        return data


    @staticmethod
    def FourierTransform(input):
        x = DataProcessing.__FourierTransform_x(input[0])
        y = DataProcessing.__FourierTransform(input[1])
        return x, y

    @staticmethod
    def nearestIDX(A, B):
        idx = np.searchsorted(B, A)
        idx = np.clip(idx, 1, len(B) -1)
        left = B[idx-1]
        right = B[idx]
        idx -= (np.abs(A-left) < np.abs(A-right))
        return idx

    @staticmethod
    def drawSchubweg(ax, V, mt):
        slimit = DataProcessing.cm2um(np.sqrt(mt*V))
        ax.plot(V, slimit, 'g')
        return

    @staticmethod
    def HechtRelation(V, d, mt, alpha, N):
        meshV, meshd = np.meshgrid(V, d)
        s = mt*meshV/meshd # schubweg
        Q = 0

        for k in range(1, N+1):
            Q = Q + (s/meshd) * np.exp(-alpha*meshd*k/N) * (1 - np.exp(-alpha*meshd/N)) * (1 - np.exp(-(meshd-k*meshd/N)/s))

        return Q

    @staticmethod
    def ModifiedHechtRelation(V, d, mt, alpha, N):
        meshV, meshd = np.meshgrid(V, d)
        s = mt*meshV/meshd # schubweg
        Q = 0

        for k in range(1, round((N+1)/2)):

            Q = Q + (s/(meshd/2)) * np.exp(-alpha*meshd*k/N) * (1 - np.exp(-alpha*meshd/N)) * (1 - np.exp(-((meshd/2)-k*meshd/N)/s))

        for k in range(round((N+1)/2), round(N+1)):

            Q = Q + (s/meshd) * np.exp(-alpha*meshd*k/N) * (1 - np.exp(-alpha*meshd/N)) * (1 - np.exp(-(meshd-k*meshd/N)/s))

        return Q

    @staticmethod
    def SphericalRadiation(S, d, g, alpha, N):
        meshS, meshd = np.meshgrid(S, d)
        Q = meshS.copy()
        Q[:] = 0
        dt = np.max(d)/N

        # for j in range(1, div+1):
        for i, j in enumerate(d):
            for k in range(1, N + 1):
                if k*dt >= 1*j:
                    break
                Q[i] = Q[i] + (np.exp(-alpha*dt*k) * (1 - np.exp(-alpha*dt)) / (((meshd[i]+g) - k*dt)**2 + meshS[i]**2)) *\
                              ((meshd[i]+g) - k * dt) / (np.sqrt(((meshd[i]+g) - k*dt)**2 + meshS[i]**2))
        return Q

    @staticmethod
    def SphericalRadiation_NegativeRefraction(S, d, g, alpha, N, g2):
        meshS, meshd = np.meshgrid(S, d)
        Q = meshS.copy()
        Q[:] = 0
        dt = np.max(d)/N

        # for j in range(1, div+1):
        for i, j in enumerate(d): # Thickness Sweep
            for k in range(1, N + 1): # Wave Propagation
                if k*dt >= 1*j:
                    break
                Q_Temp = (np.exp(-alpha*dt*k) * (1 - np.exp(-alpha*dt)) / (((meshd[i]+g) - k*dt)**2 + meshS[i]**2)) *\
                              ((meshd[i]+g) - k * dt) / (np.sqrt(((meshd[i]+g) - k*dt)**2 + meshS[i]**2))

                Ltant = g2 * meshS[i] / (meshd[i] - k*dt + g)
                idx = DataProcessing.nearestIDX(meshS[i] - Ltant, meshS[i])
                Q[i] = Q[i] + np.bincount(idx, weights = Q_Temp, minlength=len(meshS[i]))

        return Q

    @staticmethod
    def __FourierTransform(input):
        return np.fft.fftshift(np.fft.fft(input))

    @staticmethod
    def __FourierTransform_x(input):
        return np.fft.fftshift(np.fft.fftfreq(len(input), (input[1]-input[0])))

    @staticmethod
    def __rect(x, px):
        return np.where(np.abs(x) <= (px/2), 1, 0)

    @staticmethod
    def __findN(x, px):
        return (np.abs(x) - px/2)//px + 0.5

    @staticmethod
    def __findNend(x, px):
        return np.abs(x)/px

    @staticmethod
    def __Truncation(data, nPx, n, nEnd, direction='left'):

        idx = int(np.round((nEnd - n) * nPx))

        if direction == 'left':
            temp = data[:, :idx]
            data = data[:, idx:]
            return temp, data
        if direction == 'right':
            temp = data[:, data.shape[-1] - idx:]
            data = data[:, :data.shape[-1] - idx]
            return temp, data
        return

    @staticmethod
    def __Pixelation(data, nPx):
        data = np.reshape(data, (data.shape[0], -1, nPx))
        data = np.average(data, axis=-1)
        return np.repeat(data, nPx, axis=-1)



class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh, datarange):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.datarange = datarange
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        # majorlocs = self.axis.get_majorticklocs()
        majorlocs = 10 ** np.arange(np.log10(self.datarange[1]),
                                    np.log10(self.datarange[0]) - 1, -1)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

    #
    # def TextatTargetPos(self, xval, data):
    #     yidx = (np.abs(data[0]-xval)).argmin()
    #     TextHere = f"Deviation: {np.abs(100*data[1][yidx]/(self.data1[0] - data[1][yidx]))}"
    #     self.drawax.text(xval, data[1][yidx], TextHere, fontsize=fontstyle['FontSize'])
    #
    # def SaveFigure(self):
    #
    #     filepath = tkinter.filedialog.asksaveasfilename(initialdir=f"{fd}/",
    #                                                     title="Save as",
    #                                                     filetypes=(("png", ".png"),
    #                                                                ("all files", "*")))
    #     filepath = f"{filepath}.png"
    #
    #     self.fig.savefig(filepath)
    #
