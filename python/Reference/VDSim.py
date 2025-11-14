import spekpy as sp  # Import spekpy for X-ray energy spectrum calculation
import xraydb as xrdb  # x-ray material data base
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

# ------------------------------------------------------------------ #
# X-ray spectrum definition
# ------------------------------------------------------------------ #
beamQ = {'RQA3': [50, 10],  # define kVp and added Al filter thickness
         'RQA5': [70, 21],
         'RQA7': [90, 30],
         'RQA9': [120, 40]
         }


# ------------------------------------------------------------------ #
# X-ray spectrum source class
# ------------------------------------------------------------------ #
class VSource:
    def __init__(self, bQ):  # beam quality
        self.kVp = beamQ[bQ][0]  # X-ray kVp
        self.AlFilter = beamQ[bQ][1]  # added filter in mm
        self.SID = 150  # SID in cm
        self.dose = 2.5  # target dose in uGy
        # generated valuse using genSpec()
        self.k = []  # x-ray energy
        self.phiK = []  # fluence in #/mm^2
        self.hvl1 = 0  # 1st HVL
        self.q0 = 0  # photon fluence at the detector entrance in photons/mm^2

    ## Generating spectrum
    def genSpec(self):
        th = 12  # Anode angle in degrees (default: 12)
        dk = 0.1  # Spectrum bin width in keV (default: 0.5)
        physics = "legacy"  # Legacy physics mode rather than default (default: "default")
        mu_data_source = "pene"  # Penelope mu/mu_en data rather than NIST/XCOM (default: "nist")
        z = self.SID  # Point-of-interest is at a focus-to-detector-distance of 75 cm (default: 100)
        x = 0  # Point-of-interest displaced 5 cm towards cathode in anode-cathode direction (default: 0)
        y = 0  # Point-of-interest displaced -5 cm in orthogonal direction (right-hand-rule is applicable) (default: 0)
        mas = 1  # Exposure set to 2 mAs (default: 1)
        brem = True  # Whether the bremsstrahlung portion of spectrum is retained (default: True)
        char = True  # Whether the characteristic portion of spectrum is retained (default: True)
        obli = True  # Whether path-length through extrinsic filtration varies with x, y (default: True)
        # Generate unfiltered spectrum
        s = sp.Spek(kvp=self.kVp, th=th, dk=dk, physics=physics, mu_data_source=mu_data_source,
                    x=x, y=y, z=z, mas=mas, brem=brem, char=char, obli=obli)
        # Add filter
        s.filter('Glass (Plate)', 1.0).filter('Al', 4).filter('Al', self.AlFilter).filter('Air', z * 10)
        self.k, self.phiK = s.get_spectrum(edges=False, flu=True, diff=False)  # Get arrays of energy & fluence spectrum
        # Calculate metrics
        self.hvl1 = s.get_hvl1()  # Get 1st HVL
        kair = s.get_kerma()  # Get air kerma
        phi = s.get_flu()  # Get total fluence
        # dose correction
        self.mAs = self.dose / kair
        # print('mAs for', tDose, 'uGy is', round(masRatio,2),'mAs')
        self.phiK = self.phiK * self.mAs * 1e-2  # area in /mm^2
        self.q0 = phi * self.mAs * 1e-2  # photon fluence at the detector entrance in photons/mm^2

    ## ser source parameter
    def setDose(self, dose):  # set target dose in uGy
        self.dose = dose

    def setSID(self, dist):  # set source to detector distance in cm
        self.SID = dist


# ------------------------------------------------------------------ #
# X-ray detector class
# ------------------------------------------------------------------ #
class VDetector:
    def __init__(self):
        self.pxPitch = 0.140  # pixel pitch of the detector in mm
        self.pxArea = 0.140 * 0.140  # pixel area in mm^2
        ## scintillator definition
        self.element = ['Cs', 'I']
        self.weight = [0.51155, 0.48845]
        self.thickness = 400  # um
        self.density = 4.51  # g/cm^3
        self.pf = 0.90  # packing fraction of the material
        self.lightOutput = 54  # photon/keV
        self.Is = 0.90  # Swank noise factor
        self.escEff = 0.56  # light escape efficiency to the photodiode
        self.spreadCoeff = 0.5  # spread coefficient
        # calcuated based on x-ray spectrum and detector material using genAbs()
        self.muTotal = []  # attenuation coefficients
        self.abEff = []  # quantum efficiency
        # photodiode definition
        self.ff = 0.8  # photodiode fill factor
        self.pdQE = 0.8  # photodiode quantum efficiency
        self.noise = 1000  # additive noise

    ## X-ray absorption efficiency for the given energy spectrum
    def genAbs(self, source):
        # attenuation coefficients
        mu = np.empty((len(source.k), len(self.element)))
        i = 0
        for element in self.element:
            mu[:, i] = xrdb.mu_elam(element, energy=source.k * 1e3, kind='total')
            i += 1
        self.muTotal = np.dot(mu, self.weight)
        # absorption efficiency
        self.abEff = 1 - np.exp(-(self.muTotal * self.thickness * 1e-4 * self.density * self.pf))

    ## set detector parameter
    def setPp(self, pp):  # set pixel pitch
        self.pxPitch = pp
        self.pxArea = pp * pp

    def setThickness(self, thickness):  # set CsI thickness
        self.thickness = thickness

    def setPF(self, pf):  # set CsI packing fraction
        self.pf = pf

    def setSpread(self, spread):  # set CsI blur parameter
        self.spreadCoeff = spread

    def setFF(self, ff):  # set photodiode fill factor
        self.ff = ff

    def setPDQE(self, qe):  # set photodiode QE
        self.pdQE = qe

    def setNoise(self, noise):  # set additive noise
        self.noise = noise


# ------------------------------------------------------------------ #
# signal class
# ------------------------------------------------------------------ #
class VSignal:
    ## Signal Initialization
    def __init__(self, source, detector):
        n = 256  # Number of elements within the frequency range of 0 to Nyquist
        R = np.arange(2 * 256)  # Array length
        self.f = R / (detector.pxPitch * (2 * n))  # Spatial frequency cycles/mm
        self.length = np.size(self.f)
        self.signal = source.q0 * np.ones(self.length, dtype=np.float64)  # photon/mm^2
        self.wiener = source.q0 * np.ones(self.length, dtype=np.float64)  # photon/mm^2
        self.mtf = np.zeros(np.size(self.f))
        self.nnps = np.zeros(np.size(self.f))
        self.dqe = np.zeros(np.size(self.f))
        # gain and noise values
        self.g1 = 0.
        self.g2 = 0.
        self.g4 = 0.
        self.Ga = 0.
        self.noise = 0.

    ## calculating g1: X-ray absorption efficiency in the scintillator
    def quantumSelection(self, source, detector):
        # Normalized fluence spectrum
        k = source.k
        phi = source.phiK
        specNorm = phi / integrate.simps(phi, k)
        # Mean quantum efficiency
        g1 = integrate.simps(detector.abEff * specNorm, k)
        self.g1 = g1
        g1Std = np.sqrt(g1 * (1 - g1))
        # stochastic gain
        stochasticGain(self, g1, g1Std)

    ## calculating g2: Secondary quanta amplification gain in the scintillator
    def quantumGain(self, source, detector):
        k = source.k
        phi = source.phiK
        QE = detector.abEff
        LO = detector.lightOutput
        Is = detector.Is

        g2A = integrate.simps(QE * LO * k * phi, k) / integrate.simps(QE * phi, k)
        g2B = detector.escEff
        g2 = g2A * g2B
        self.g2 = g2
        eps = g2 / Is - g2 - 1
        g2Std = np.sqrt(g2 * (1 + eps))
        stochasticGain(self, g2, g2Std)

    ## calculating quantum blur in the scintillator
    def quantumBlur(self, detector):
        H = detector.spreadCoeff
        f = self.f
        # optical spread function
        # osf = 1 / (1 + H * f + H * f ** 2 + H ** 2 * f ** 3)
        # osf = 1 / (1 + H * ( f + f**2 ) )
        osf = 1 / (1 + H * (f ** 2))
        quantumScatter(self, osf)

    ## calculating optical coupling to the photodiode
    def opticalCouple(self, detector):
        # coupling efficiency from the detector
        g4a = detector.ff
        g4b = detector.pdQE
        g4 = g4a * g4b
        self.g4 = g4
        g4Std = np.sqrt(g4a * (1 - g4a) + g4b * (1 - g4b))
        # Optical coupling is represented by a stochastic gain stage
        stochasticGain(self, g4, g4Std)

    ## integration of quanta by photodiode
    def pxIntegration(self, source, detector):
        # Apply deterministic blur using the pixel aperture function
        # t = np.abs(np.sinc(detector.pxPitch * detector.ff * self.f))
        t = np.abs(np.sinc(detector.pxPitch * self.f))
        deterministicBlur(self, t)

        # Integrated Signal and Wiener spectrum
        self.signal = self.signal * detector.pxArea  # signal becomes unitless
        self.wiener = self.wiener * (detector.pxArea ** 2)  # noise becomes in mm^2

        # Modulation Transfer Function
        mtf = self.signal / self.signal[0]

        # Aliased Noise Power Spectrum up to Nyquist frequency
        reSample(self)

        # Additive electronic noise
        addNoise = detector.noise  # / detector.pxArea
        nps = self.wiener[0:int(self.length / 2)] + ((addNoise ** 2) * detector.pxArea)
        f2 = self.f[0:int(self.length / 2)]
        self.mtf = mtf[0:int(self.length / 2)]

        # Normalized NPS (Noise Power Spectrum)
        self.nps = nps
        self.nnps = nps / (self.signal[0] ** 2)
        self.f = f2
        self.length = int(self.length / 2)
        self.noise = integrate.simps(np.sqrt(nps), f2) * 2
        # self.noise = np.sqrt(integrate.simps(nps, f2))

        # Detective quantum efficiency
        q0 = source.q0
        self.dqe = (self.mtf ** 2) / self.nnps / q0

        self.Ga = self.signal[0]


# ------------------------------------------------------------------ #
# define signal process
# ------------------------------------------------------------------ #

## stochastic gain process
def stochasticGain(signal, gainMean, gainStd):
    signalIn = signal.signal
    wienerIn = signal.wiener
    signal.signal = gainMean * signalIn
    signal.wiener = (gainMean ** 2) * wienerIn + (gainStd ** 2) * signalIn


## quantum scatter process
def quantumScatter(signal, t):
    signalIn = signal.signal
    wienerIn = signal.wiener
    signal.signal = t * signalIn
    signal.wiener = ((t ** 2) * (wienerIn - signalIn)) + signalIn


## deterministic blur process
def deterministicBlur(signal, t):
    signalIn = signal.signal
    wienerIn = signal.wiener
    signal.signal = t * signalIn
    signal.wiener = (t ** 2) * wienerIn


## aliasing resampling process
def reSample(signal):
    wienerIn = signal.wiener
    reverse = np.flip(wienerIn)
    signal.wiener = wienerIn + reverse


# ------------------------------------------------------------------ #
# present results
# ------------------------------------------------------------------ #

## print source parameter and spectrum
def showSource(source):
    energy = source.k
    fluence = source.phiK
    hvl1 = source.hvl1
    q0 = source.q0
    mAs = source.mAs
    dose = source.dose
    # Print metrics
    print('****  Defined source characteristics  ****')
    print('HVL1:', '{:.2f}'.format(hvl1), 'mm Al')
    print('Fluence:', "{:e}".format(q0 / dose), '/mm^2/uGy')
    print('mAs:', "{:.2f}".format(mAs), 'mAs')
    print('dose:', "{:.2f}".format(dose), 'uGy')
    print('q0:', "{:e}".format(q0), '/mm^2')
    print('  ')
    # Plot the x-ray spectrum
    plt.plot(energy, fluence)
    plt.xlabel('Energy  [keV]')
    plt.ylabel('Differential fluence  [mm$^{-2}$ keV$^{-1}$]')
    plt.title('X-ray spectrum @ RQA5')
    plt.show()


## plot absorption curve
def plotAbs(source, detector):
    plt.plot(source.k, detector.abEff)
    plt.xlabel('Energy  [keV]')
    plt.ylabel('Absorption Efficiency')
    plt.ylim(0, 1)
    plt.title('X-ray absorption efficiency')
    plt.show()


## print gain of each stage
def showGains(signal):
    print('****  quantum gain of each stage  ****')
    print('g1:', '{:.2f}'.format(signal.g1))
    print('g2:', '{:.2f}'.format(signal.g2))
    print('g4:', '{:.2f}'.format(signal.g4))
    print('  ')


## print Signal
def showSignal(signal):
    print('****  Signal / sensitivity  ****')
    print('Signal(e-)/pixel:', '{:.2f}'.format(signal.Ga))
    print('Noise(e-)/pixel:', '{:.2f}'.format(signal.noise))


## plot MTF curve
def plotMTF(signal):
    plt.plot(signal.f, signal.mtf, color='black', label='MTF_System')
    plt.xlabel('Spatial frequency (/mm)')
    plt.ylabel('MTF')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlim(0, signal.f[-1])
    plt.grid(which='major', axis='both', ls='-')
    plt.show()


## plot DQE curve
def plotDQE(signal):
    plt.plot(signal.f, signal.dqe, color='red', label='DQE')
    plt.xlabel('Spatial frequency (/mm)')
    plt.ylabel('DQE')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlim(0, signal.f[-1])
    plt.grid(which='major', axis='both', ls='-')
    plt.show()