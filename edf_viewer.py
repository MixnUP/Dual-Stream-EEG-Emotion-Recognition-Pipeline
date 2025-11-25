import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from mne.viz import plot_raw_psd
import io
from scipy import signal

# Set page config
st.set_page_config(page_title="EDF Viewer", layout="wide")

def apply_filters(raw, l_freq=1.0, h_freq=45.0, notch_freqs=[50.0, 60.0]):
    """Apply filters to raw data and return a copy"""
    filtered = raw.copy()
    
    # Apply bandpass filter
    if l_freq is not None or h_freq is not None:
        filtered = filtered.filter(
            l_freq=l_freq, 
            h_freq=h_freq,
            method='fir',
            phase='zero-double',
            verbose=False
        )
    
    # Apply notch filters
    if notch_freqs:
        for freq in notch_freqs:
            filtered = filtered.notch_filter(
                freqs=freq,
                method='fir',
                verbose=False
            )
    
    return filtered

def load_edf(uploaded_file):
    """Load EDF file from uploaded file object"""
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Load the EDF file
        raw = mne.io.read_raw_edf(tmp_path, preload=True)
        
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        return raw
    except Exception as e:
        st.error(f"Error loading EDF file: {str(e)}")
        return None

def plot_eeg_data(ax, data, times, channels, title, scale=1.0):
    """Helper function to plot EEG data with adjustable scaling"""
    n_channels = len(channels)
    ch_ranges = np.ptp(data, axis=1)
    base_spacing = 2 * np.max(ch_ranges) if np.max(ch_ranges) > 0 else 1.0
    spacing = base_spacing * scale  # Apply scaling factor
    
    for i, (ch_data, ch_name) in enumerate(zip(data, channels)):
        offset = i * spacing
        ax.plot(times, ch_data - ch_data.mean() + offset, label=ch_name, linewidth=0.8)
    
    # Set y-ticks to show channel names
    y_ticks = [i * spacing for i in range(n_channels)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(channels)
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def main():
    st.title("EEG Data Viewer with Filtering")
    
    # Sidebar controls
    st.sidebar.header("Filter Settings")
    
    # Filter parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        l_freq = st.number_input("Low Cutoff (Hz)", min_value=0.0, value=1.0, step=0.5)
    with col2:
        h_freq = st.number_input("High Cutoff (Hz)", min_value=1.0, value=45.0, step=0.5)
    
    notch_enabled = st.sidebar.checkbox("Enable Notch Filter", value=True)
    notch_freqs = [50.0, 60.0] if notch_enabled else []
    
    # File upload
    st.sidebar.header("File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload EDF file", type=["edf", "bdf"])
    
    if uploaded_file is not None:
        with st.spinner('Loading EDF file...'):
            raw = load_edf(uploaded_file)
        
        if raw is not None:
            # Display file info
            st.sidebar.subheader("File Information")
            st.sidebar.write(f"Channels: {len(raw.ch_names)}")
            st.sidebar.write(f"Sampling rate: {raw.info['sfreq']} Hz")
            st.sidebar.write(f"Duration: {raw.times[-1]:.2f} seconds")
            
            # Channel selection
            st.sidebar.subheader("Display Options")
            channels = st.sidebar.multiselect(
                "Select channels to display",
                options=raw.ch_names,
                default=raw.ch_names[:min(8, len(raw.ch_names))]
            )
            
            if not channels:  # If no channels selected, use first 8
                channels = raw.ch_names[:min(8, len(raw.ch_names))]
            
            # Time range selection
            max_time = raw.times[-1]
            time_range = st.sidebar.slider(
                "Time range (s)",
                min_value=0.0,
                max_value=float(max_time),
                value=(0.0, min(10.0, max_time)),
                step=0.1
            )
            
            # Plot options
            st.sidebar.subheader("Display Settings")
            show_psd = st.sidebar.checkbox("Show Power Spectral Density", value=False)
            show_difference = st.sidebar.checkbox("Show Difference (Raw - Filtered)", value=False)
            y_scale = st.sidebar.slider("Vertical Scale", 0.1, 5.0, 1.0, 0.1, 
                                      help="Adjust the vertical spacing between channels")
            
            # Apply filters
            filtered_raw = apply_filters(
                raw.copy(),
                l_freq=l_freq if l_freq > 0 else None,
                h_freq=h_freq if h_freq > 0 else None,
                notch_freqs=notch_freqs
            )
            
            # Get data for selected channels and time range
            start_idx = int(time_range[0] * raw.info['sfreq'])
            stop_idx = int(time_range[1] * raw.info['sfreq'])
            
            # Get data for plotting
            data, times = raw[channels, start_idx:stop_idx]
            data_filt, _ = filtered_raw[channels, start_idx:stop_idx]
            
            if show_psd:
                # Create separate figures for raw and filtered PSDs with more square dimensions
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Raw PSD
                raw.pick(channels).compute_psd().plot(axes=ax1, show=False, color='blue')
                ax1.set_title('Raw PSD')
                
                # Raw time series
                plot_eeg_data(ax2, data, times, channels, 'Raw EEG Data', y_scale)
                
                plt.tight_layout()
                st.pyplot(fig1)
                
                # Create a new figure for filtered data with matching dimensions
                fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Filtered PSD
                filtered_raw.pick(channels).compute_psd().plot(axes=ax3, show=False, color='orange')
                ax3.set_title('Filtered PSD')
                
                # Filtered time series
                plot_title = (f'Filtered EEG Data ({l_freq}-{h_freq} Hz, ' + 
                            f'Notch: {notch_freqs if notch_enabled else "Off"}, ' +
                            f'Scale: {y_scale}x')
                plot_eeg_data(ax4, data_filt, times, channels, plot_title, y_scale)
                
                plt.tight_layout()
                st.pyplot(fig2)
                
            else:
                # Create separate figures for raw and filtered time series with more square dimensions
                fig1, ax1 = plt.subplots(figsize=(12, 8))
                plot_eeg_data(ax1, data, times, channels, 'Raw EEG Data', y_scale)
                plt.tight_layout()
                st.pyplot(fig1)
                
                # Second figure for filtered data or difference with matching dimensions
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                
                if show_difference:
                    # Show difference between raw and filtered
                    diff = data - data_filt
                    plot_eeg_data(ax2, diff, times, channels, 
                                f'Difference (Raw - Filtered) - Scale: {y_scale}x', y_scale)
                else:
                    # Show filtered data
                    plot_title = (f'Filtered EEG Data ({l_freq}-{h_freq} Hz, ' + 
                                f'Notch: {notch_freqs if notch_enabled else "Off"}, ' +
                                f'Scale: {y_scale}x')
                    plot_eeg_data(ax2, data_filt, times, channels, plot_title, y_scale)
                
                plt.tight_layout()
                st.pyplot(fig2)
            
            # Add channel information
            st.subheader("Channel Information")
            info_data = []
            for ch in channels:
                idx = raw.ch_names.index(ch)
                ch_type = raw.get_channel_types()[idx]
                info_data.append({
                    "Channel": ch, 
                    "Type": ch_type, 
                    "Sampling Rate": f"{raw.info['sfreq']} Hz"
                })
            
            st.table(info_data)
            
            # Add download button for the displayed data
            if st.sidebar.button("Export Displayed Data as CSV"):
                # Get the data for selected channels and time range
                data, times = raw[channels, start_idx:stop_idx]
                
                # Create a DataFrame
                import pandas as pd
                df = pd.DataFrame(data.T, columns=channels)
                df["Time (s)"] = times + time_range[0]
                
                # Convert to CSV
                csv = df.to_csv(index=False).encode('utf-8')
                
                # Create download button
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{uploaded_file.name.split('.')[0]}_export.csv",
                    mime="text/csv"
                )
    else:
        st.info("Please upload an EDF file using the sidebar.")
        
        # Add some example usage instructions
        st.markdown("""
        ### How to use this EDF Viewer:
        1. Click on "Browse files" in the sidebar to upload an EDF file
        2. Select which channels to display from the sidebar
        3. Adjust the time range using the slider
        4. Toggle between time series and PSD view
        5. Adjust filter settings as needed
        6. Export the displayed data as CSV if needed
        
        ### Supported formats:
        - EDF/EDF+ files (.edf)
        - BDF files (.bdf)
        """)

if __name__ == "__main__":
    main()
