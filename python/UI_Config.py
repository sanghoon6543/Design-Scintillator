from tkinter import *
from tkinter.ttk import Combobox

class UI_tkinter:
    @staticmethod
    def MakeWindow(window, title, fw, fh, resize=False, background='#FFFFFF'):

        window.title(title)
        window.config(background=background)
        window.geometry(f"{fw}x{fh}")
        window.resizable(resize, resize)
        return window

    @staticmethod
    def MakeFrameLabel(window, fw, fh, col, row, text="", padx=10, pady=10, ftstyle='Calibri', ftsize=18):

        FrameLabel = LabelFrame(window, width=fw, height=fh, text=text, font=(f"{ftstyle} {ftsize}"))
        FrameLabel.grid(column=col, row=row, padx=padx, pady=pady)
        return FrameLabel

    @staticmethod
    def MakeFrame(window, fw, fh, row, col, colspan, bg='white'):

        frame = Frame(window, bg=bg, width=fw, height=fh)
        frame.grid(column=col, row=row, columnspan=colspan, padx=10, pady=10)
        return frame

    @staticmethod
    def UI_Labels(frame, t="", row=0, col=0, rowspan=1, colspan=1, width=14, height=2, padx=2, pady=2, ftstyle='Calibri', ftsize=10):

        L = Label(frame, width=width, height=height, text=t, relief="ridge", font=(f"{ftstyle} {ftsize}"))
        L.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan, padx = padx, pady= pady)

    @staticmethod
    def UI_InputEntry(frame, t, row=0, col=1, rowspan=1, colspan=1, width=20, padx=2, pady=2, ftstyle='Calibri', ftsize=10):

            # if type(t) is str:
                EntryAddress = Entry(frame, width=width,  textvariable="", relief="ridge", font=(f"{ftstyle} {ftsize}"))
                EntryAddress.grid(row=row, column = col, rowspan=rowspan, columnspan=colspan, padx=padx, pady=pady)
                EntryAddress.insert(0, t)

                return EntryAddress

    @staticmethod
    def UI_CBox(frame, t, row=0, col=1, rowspan=1, colspan=1, width=20, padx=2, pady=2, ftstyle='Calibri', ftsize=10):
        CBoxAddress = Combobox(frame, width=width, textvariable="", state="readonly", values=t, font=(f"{ftstyle} {ftsize}"))
        CBoxAddress.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan, padx=padx, pady=pady)
        CBoxAddress.set(t[0])
        return CBoxAddress

    @staticmethod
    def UI_Button(frame, t, row, col, rowspan=1, colspan=1, width=20, height=2, padx=2, pady=2, ftstyle='Calibri', ftsize=10):
        EntryAddress = Button(frame, width=width, height=height, text=t, relief="raised", font=(f"{ftstyle} {ftsize}"))
        EntryAddress.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan, padx=padx, pady=pady)
        return EntryAddress
