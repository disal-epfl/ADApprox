# MIT License
#
# Copyright (c) 2025 Nicolaj Bösel-Schmid
# Contact: nicolaj.schmid@epfl.ch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
import copy

from typing import Dict, List
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, periodogram
from matplotlib import pyplot as plt





class SignalProcessing():
    def __init__(
        self,
        data: Dict[str, np.ndarray],
        print_stats: bool = False,
    ):
        # Parameters
        self.print_stats = print_stats

        # set the data
        self._data: Dict[str, np.ndarray] = {}
        self.set_data(
            data=data,
        )

    def __len__(
        self,
    ):
        """
        Get the number of samples.
        Returns:
            n_samples (int): The number of samples.
        """
        if len(self._data) == 0:
            return 0
        return self._data[next(iter(self._data))].shape[0]

    def get_data(
        self,
        keys: list = None,
        copy_data: bool = True,
    ):
        """
        Get the data.
        Args:
            keys (list): The keys to get the data. If None, then all data is returned.
            copy_data (bool): Copy the data.
        Returns:
            data (Dict[str, np.ndarray]): The data. Key is the name of the data.
                                            And the value is the data.
        """
        if keys is not None:
            data = {key: self._data[key] for key in keys}
        else:
            data = self._data

        if copy_data:
            return copy.deepcopy(data)
        return data
    
    def set_data(
        self,
        data: Dict[str, np.ndarray],
        copy_data: bool = True,
    ):
        """
        Set the data.
        Args:
            data (Dict[str, np.ndarray]): The data. Key is the name of the data.
            copy_data (bool): Copy the data.
        """
        for key in data.keys():
            if copy_data:
                self._data[key] = np.copy(data[key])
            else:
                self._data[key] = data[key]

    def remove_outliers(
        self,
        data_lims: Dict[str, tuple],
    ):
        """
        Remove outliers from the data.
        Args:
            data_lims (Dict[str, tuple]): The limits for the data.
                                          Key is the name of the data.
                                          Value is the limits (min, max).
                                          If data[key] is multi-dimensional, then
                                          min/max should be a numpy array.
        """
        data = self.get_data(
            copy_data=False,
        )

        mask = np.ones(len(self), dtype=bool)
        mask_dict: Dict[str, np.ndarray] = {}
        for key, lims in data_lims.items():
            if key == "length":
                mask_ = np.zeros(len(self), dtype=bool)
                mask_[lims[0]:lims[1]] = True
                mask_dict[key] = mask_
            else:
                if type(lims[0]) is float:
                    mask_ = (data[key] >= lims[0]) & (data[key] <= lims[1])
                elif type(lims[0]) is np.ndarray and lims[0].ndim == 1:
                    mask_ = np.all(data[key] >= lims[0], axis=-1) & np.all(data[key] <= lims[1], axis=-1)
                else:
                    raise ValueError("Data limits not understood.")
                mask_dict[key] = mask_
            mask = mask & mask_dict[key]

        for key, arr in data.items():
            data[key] = arr[mask]

        self._print_stats_mask(
            fct_name="crop",
            mask=mask,
            mask_dict=mask_dict,
        )

        self.set_data(
            data=data,
            copy_data=False,
        )
    
    def remove_duplicates(
        self,
        keys: tuple,
    ):
        """
        Remove duplicate time values.
        Args:
            keys (tuple): The keys to remove duplicates.
        """
        data = self.get_data(
            copy_data=False,
        )

        mask = np.ones(len(self), dtype=bool)
        mask_dict: Dict[str, np.ndarray] = {}
        for key in keys:
            _, unique_indices = np.unique(data[key], return_index=True, axis=0)
            mask_ = np.zeros(len(self), dtype=bool)
            mask_[unique_indices] = True
            mask = mask & mask_
            mask_dict[key] = mask_

        for key, arr in data.items():
            data[key] = arr[mask]

        self._print_stats_mask(
            fct_name="remove_duplicates",
            mask=mask,
            mask_dict=mask_dict,
        )

        self.set_data(
            data=data,
            copy_data=False,
        )

    def apply_fcts(
        self,
        fcts_dict: Dict[str, callable],
    ):
        """
        Apply functions to the data.
        Args:
            fcts_dict (Dict[str, callable]): The functions to apply and 
                                            the keys to apply them to.
        """
        data = self.get_data(
            keys=fcts_dict.keys(),
        )

        for key, fct in fcts_dict.items():
            data[key] = fct(data[key])

        self.set_data(
            data=data,
            copy_data=False,
        )

    def apply_offsets(
        self,
        offsets_dict: Dict[str, float],
    ):
        """
        Apply offset to the signals.
        Args:
            offsets_dict (Dict[str, float]): The offsets to remove from corresponding signals.
        """
        data = self.get_data(
            keys=offsets_dict.keys(),
            copy_data=False,
        )

        for key, offset in offsets_dict.items():
            data[key] += offset

        self.set_data(
            data=data,
        )

    def apply_interpolation(
        self,
        interpolation_key: str,
        frequency: float,
    ):
        """
        Interpolate the samples to a given frequency of the chosen signal.
        Args:
            key (str): The key of the signal to interpolate.
            frequency (float): The frequency to interpolate to.
        """
        data = self.get_data(
            copy_data=False,
        )

        time = data[interpolation_key].copy()
        time_new = np.arange(time[0], time[-1], 1/frequency)
        time_new = np.clip(time_new, time[0], time[-1]) # correct numerical errors
        for key, arr in data.items():
            f = interp1d(time, arr, kind='linear', axis=0)
            data[key] = f(time_new)

        self.set_data(
            data=data,
            copy_data=False,
        )

    def derivative(
        self,
        signal_keys: List[str],
        avg_window: int = 1,
    ):
        """
        Calculate the derivative of the signals.
        Args:
            signal_keys (List[str]): The keys of the signals.
        """
        data = self.get_data(
            keys=signal_keys+['time'],
            copy_data=False,
        )

        data_new = {}
        for key in signal_keys:
            grad = np.gradient(data[key], data['time'])

            if avg_window > 1:
                grad = np.convolve(grad, np.ones(avg_window)/avg_window, mode='same')

            data_new[key+'_grad'] = grad

        self.set_data(
            data=data_new,
        )

    def time_shift(
        self,
        signal_keys: List[str],
        delay: float,
    ):
        """
        Shift the signals in time.
        Args:
            signal_keys (List[str]): The keys of the signals.
            delay (float): The delay in seconds to shift the signals.
        """
        data = self.get_data()

        # shift the indicated signals
        shift = int(delay / (self.get_data()['time'][1] - self.get_data()['time'][0]))
        for key in signal_keys:
            data[key] = data[key][shift:]

        # shorten all other signals
        for key in data.keys():
            data[key] = data[key][:len(data[signal_keys[0]])]

        self.set_data(
            data=data,
        )

    def filter_high_frequency(
        self,
        signal_keys: list,
        cutoff_freq: float,
    ):
        """
        Remove high frequency noise from the signal.
        Args:
            signal_keys (list): The keys of the signals.
            cutoff_freq (float): The cutoff frequency.
        """
        data = self.get_data(
            keys=signal_keys+['freq'],
        )

        mask_ = np.abs(data['freq']) > cutoff_freq
        for key in signal_keys:
            arr = np.copy(data[key])
            arr[mask_] = 0
            data[key+'_cutoff'] = arr

        self.set_data(
            data=data,
        )

    def filter_savgol(
        self,
        signal_key: str,
        window_length: int,
        polyorder: int = 3,
    ):
        """
        Apply the Savitzky-Golay filter to the signal.
        Args:
            signal_key (str): The key of the signal.
            window_length (int): The length of the filter window.
            polyorder (int): The order of the polynomial.
        """
        data = self.get_data(
            keys=[signal_key],
            copy_data=False,
        )
        
        signal_savgol = savgol_filter(
            x=data[signal_key], 
            window_length=window_length,
            polyorder=polyorder,
            axis=0,
        )

        self.set_data(
            data={
                signal_key+'_savgol': signal_savgol,
            },
        )

    def filter_ma(
        self,
        signal_keys: List[str],
        window_length: int,
    ):
        """
        Apply a moving average to signals.
        Args:
            signal_keys
            window_length
        """
        data = self.get_data(
            keys=signal_keys,
        )

        df = pd.DataFrame(data)
        for key in signal_keys:
            df[key+'_ma'] = np.convolve(df[key], np.ones(window_length)/window_length, mode='same')
        data = df.to_dict(orient='list')

        self.set_data(
            data=data,
        )

    def filter_ema(
        self,
        signal_keys: List[str],
        alphas: List[float],
    ):
        """
        Apply an exponential moving average to signals.
        Args:
            signal_keys
        """
        data = self.get_data(
            keys=signal_keys,
        )

        df = pd.DataFrame(data)
        for key, alpha in zip(signal_keys, alphas):
            df[key+'_ema'] = df[key].ewm(alpha=alpha, adjust=False).mean()
        data = df.to_dict(orient='list')

        self.set_data(
            data=data,
        )

    def deconv_1_order(
        self,
        signal_keys: str,
        filter_gain: float,
        time_cosntant: float,
    ):
        """
        Deconvolution using a 1nd order filter.
        The transfer function of the sensor is defined as follows:
            H(s) = K / (s*tau + 1) where K = gain, tau = time constant
        Assume the time steps are equal (apply first inerpolate_time).
        Args:
            signal_keys (str): The key of the signal.
            filter_gain (float): The gain of the filter.
            time_cosntant (float): The time constant of the filter.
        """
        data = self.get_data(
            keys=signal_keys+['time'],
        )
        freq = fftfreq(len(self), d=(data['time'][1] - data['time'][0]))

        # Calculate the filter in the frequency domain
        transfer_function = filter_gain / ((2*np.pi*1j*time_cosntant*freq + 1) + 1e-10)
        
        for key in signal_keys:
            signal_fft = fft(data[key])
            signal_fft_deconv = (signal_fft * np.conj(transfer_function)) / (np.abs(transfer_function)**2 + 1e-10)
            signal_deconv = np.real(ifft(signal_fft_deconv))

            data[key+'_fft'] = signal_fft
            data[key+'_deconv1'] = signal_deconv

        data['freq'] = freq
        self.set_data(
            data=data,
        )

    def deconv_2_order(
        self,
        signal_keys: str,
        filter_gain: float,
        filter_pole1: float,
        filter_pole2: float,
    ):
        """
        Deconvolution using a 2nd order filter.
        The transfer function of the sensor is defined as follows:
            H(s) = K / ((s + c1)(s + c2)) where K = gain, c1, c2 = poles
        Assume the time steps are equal (apply first inerpolate_time).
        Args:
            signal_keys (str): The key of the signal.
            filter_gain (float): The gain of the filter.
            filter_pole1 (float): The first pole of the filter.
            filter_pole2 (float): The second pole of the filter.
        """
        data = self.get_data(
            keys=signal_keys+['time'],
        )
        freq = fftfreq(len(self), d=(data['time'][1] - data['time'][0]))

        # Calculate the filter in the frequency domain
        transfer_function = filter_gain / ((2*np.pi*1j*freq + filter_pole1) * (2*np.pi*1j*freq + filter_pole2))
        
        for key in signal_keys:
            signal_fft = fft(data[key])
            signal_fft_deconv = (signal_fft * np.conj(transfer_function)) / (np.abs(transfer_function)**2 + 1e-10)
            signal_deconv = np.real(ifft(signal_fft_deconv))

            data[key+'_fft'] = signal_fft
            data[key+'_deconv2'] = signal_deconv

        data['freq'] = freq
        self.set_data(
            data=data,
        )

    def power_spectrum(
        self,
        signal_key: str,
        nois_meas: np.ndarray,
    ):
        """
        Calculate the power spectrum density of the signal and the noise.
        Args:
            signal_key (str): The key of the signal.
            nois_meas (np.ndarray): Blank measurmement.
        """
        data = self.get_data(
            keys=[signal_key, 'time'],
            copy_data=False,
        )

        # Set the noise measurement to the signal length
        if len(self) > nois_meas.shape[0]:
            nois_meas = np.hstack((nois_meas, np.zeros(len(self) - nois_meas.shape[0])))
        else:
            nois_meas = nois_meas[:len(self)]

        # Calculate the signal and noise FFT and power spectrum
        meas_freq = 1/(data['time'][1]-data['time'][0])
        signal_fft, signal_power_spectrum = periodogram(data[signal_key], fs=meas_freq, scaling='spectrum', return_onesided=False)
        noise_fft, noise_power_spectrum = periodogram(nois_meas, fs=meas_freq, scaling='spectrum', return_onesided=False)

        self.set_data(
            data={
                signal_key+'_fft': signal_fft,
                signal_key+'_fft_ps': signal_power_spectrum,
                signal_key+'_noise_fft': noise_fft,
                signal_key+'_noise_fft_ps': noise_power_spectrum,
            },
        )

    def fft(
        self,
        signal_keys: List[str],
    ):
        """
        Calculate the FFT of the signals.
        Args:
            keys (List[str]): The keys of the signals.
        """
        data = self.get_data(
            keys=signal_keys+['time'],
            copy_data=False,
        )

        signal_ffts = {}
        for key in signal_keys:
            signal_ffts[key+'_fft'] = fft(data[key])

        freq = fftfreq(len(self), d=(data['time'][1] - data['time'][0]))

        self.set_data(
            data={
                **signal_ffts,
                'freq': freq,
            },
        )

    def ifft(
        self,
        signal_keys: List[str],
    ):
        """
        Calculate the FFT of the signals.
        Args:
            keys (List[str]): The keys of the signals.
        """
        data = self.get_data(
            keys=signal_keys,
            copy_data=False,
        )

        signal_iffts = {}
        for key in signal_keys:
            signal_iffts[key+'_ifft'] = ifft(data[key])

        self.set_data(
            data=signal_iffts,
        )

    def wiener_deconvolution(
        self,
        signal_key: str,
        filter_gain: float,
        filter_pole1: float,
        filter_pole2: float,
    ):
        """
        Wiener deconvolution applies a deconvolution filter to the signal:
            G(f) = H_conj(f) / (|H(f)|^2 + 1/SNR(f))
        The transfer function of the sensor is defined as follows:
            H(s) = K / ((s + c1)(s + c2)) where K = gain, c1, c2 = poles
        Assume the time steps are equal (apply first inerpolate_time).
        """
        data = self.get_data(
            keys=[signal_key+'_fft', signal_key+'_fft_ps', signal_key+'_noise_fft_ps', 'freq'],
            copy_data=False,
        )
        signal_fft = data[signal_key+'_fft']
        freq = data['freq']
        signal_ps = data[signal_key+'_fft_ps']
        noise_ps = data[signal_key+'_noise_fft_ps']

        # Calculate the filter in the frequency domain
        transfer_function = filter_gain / ((2*np.pi*1j*freq + filter_pole1) * (2*np.pi*1j*freq + filter_pole2))
        # filter = (np.conj(transfer_function)*data[signal_key+'_fft_ps']) / (np.abs(transfer_function)**2 + 1e-10) # TODO: add SNR
        filter = (np.conj(transfer_function) * signal_ps) / (np.abs(transfer_function)**2 * signal_ps + noise_ps)

        # Calculate deconvoluted signal
        signal_fft_deconv = signal_fft * filter
        signal_deconv = np.real(ifft(signal_fft_deconv))

        self.set_data(
            data={
                signal_key+'_deconv': signal_deconv,
            },
        )

    def _print_stats_mask(
        self,
        fct_name: str,
        mask: np.ndarray,
        mask_dict: Dict[str, np.ndarray],
    ):
        if not self.print_stats:
            return
        
        print(f"SignalProcessing: {fct_name}")
        print(f"Keep totally {mask.sum()}/{mask.size} = {mask.sum()/mask.size*100:.2f}%")
        for key, val in mask_dict.items():
            print(f"{key}: {val.sum()}/{val.size} = {val.sum()/val.size*100:.2f}%")
        print("")

    def plot_data(
        self,
        ax: plt.Axes,
        y_axis_keys: list,
        x_axis_key: str = 'time',
        labels: list = None,
        y_axis_label: str = None,
        x_axis_label: str = 'Time [s]',
        title: str = None,
        twinx: bool = False,
        fill_area_mask: np.ndarray = None,
    ):
        """
        Plot the data.
        Args:
            ax (plt.Axes): The axes to plot the data.
            y_axis_keys (list): The keys to plot on the y-axis.
            x_axis_key (str): The key to plot on the x-axis.
            labels (list): The labels for the keys.
            y_label (str): The y label.
            title (str): The title.
            twinx (bool): Plot the data on the same x-axis but different y-axis.
        Returns:
            ax (plt.Axes): The axes with the plot.
        """
        data = self.get_data(
            copy_data=False,
        )

        if labels is None:
            labels = y_axis_keys

        
        # First y-axis
        if not twinx:
            for i, key in enumerate(y_axis_keys):
                ax.plot(data[x_axis_key], data[key], label=labels[i])

                if fill_area_mask is not None:
                    ax.fill_between(data[x_axis_key], np.min(data[key]), np.max(data[key]), 
                                    where=fill_area_mask, color='green', alpha=0.3)

            ax.legend()
            ax.set_xlabel(x_axis_label)
            if y_axis_label is not None:
                ax.set_ylabel(y_axis_label)
            if title is not None:
                ax.set_title(title)

            

            return ax

        # Second y-axis
        ax2 = ax.twinx()
        for i, key in enumerate(y_axis_keys):
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][-i-1]
            ax2.plot(data[x_axis_key], data[key], label=labels[i], color=color)
        if y_axis_label is not None:
            ax2.set_ylabel(y_axis_label)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc=0)


        return ax2