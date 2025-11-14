from VDSim import *

# ------------------------------------------------------------------ #
# simulation
# ------------------------------------------------------------------ #

## source setting
mySource = VSource('RQA5')
mySource.setDose(0.2)
mySource.setSID(150)
mySource.genSpec()

## detector setting
myDetector = VDetector()
myDetector.setPp(0.100)
myDetector.setThickness(200)
myDetector.setPF(0.8)
myDetector.setSpread(0.6)
myDetector.setFF(0.7)
myDetector.setPDQE(0.8)
myDetector.setNoise(800)
myDetector.genAbs(mySource)

## cascade process
mySignal = VSignal(mySource, myDetector)
mySignal.quantumSelection(mySource, myDetector)
mySignal.quantumGain(mySource, myDetector)
mySignal.quantumBlur(myDetector)
mySignal.opticalCouple(myDetector)
mySignal.pxIntegration(mySource, myDetector)

## show the results
showSource(mySource)
plotAbs(mySource, myDetector)
showGains(mySignal)
showSignal(mySignal)
plotMTF(mySignal)
plotDQE(mySignal)