# TODO: Future Improvements

## Short Term

- [ ] Implement real hardware playback/recording with sounddevice
- [ ] Add calibration file support for measurement microphones
- [ ] Implement multi-channel simultaneous measurement
- [ ] Add real-time FFT monitoring during measurement
- [ ] Derive VENU360 control protocol via packet capture (no public spec — see dsp/venus360.py)
- [ ] Polish session report export (before/after coverage maps, algorithm comparison, recommendations)

## Medium Term

- [ ] Implement Dante Controller API integration
- [ ] Add AES67 RTP stream support
- [ ] Build automated venue calibration wizard
- [ ] Add support for non-rectangular room geometries
- [ ] Implement beamforming optimization for array speakers
- [ ] Add GPU acceleration for FIR convolution
- [ ] Implement all-pass filter design for phase alignment

## Long Term

- [ ] Linux platform support
- [ ] macOS platform support
- [ ] VST3/AU plugin integration
- [ ] Web-based remote control interface
- [ ] Machine learning model for reflection pattern prediction
- [ ] Automated microphone position optimization
- [ ] Cloud-based optimization database for venue profiles
- [ ] Real-time adaptive DSP during performances

## Technical Debt

- [ ] Add property-based testing with hypothesis
- [ ] Add benchmark suite for optimization algorithms
- [ ] Complete type annotations across all modules
- [ ] Add performance profiling integration
- [ ] Create CI/CD pipeline configuration
