from scapy.all import *
import pandas as pd
import numpy as np

def extract_features(packets, duration=10):
    """
    Extract features from a list of packets over a specified duration (seconds).
    Features match CICIDS2017 dataset.
    """
    if not packets:
        print("Error: No packets provided to extract_features.")
        return None
    
    # Initialize variables
    flow_duration = duration * 1000000  # Convert to microseconds
    fwd_packets = 0
    bwd_packets = 0
    fwd_packet_lengths = []
    bwd_packet_lengths = []
    fwd_iat = []
    bwd_iat = []
    packet_lengths = []
    
    # Use the first packet to determine flow direction (if any packets exist)
    try:
        start_time = packets[0].time
        if IP not in packets[0]:
            print("Warning: First packet has no IP layer. Skipping flow direction setup.")
            src_ip = None
            dst_ip = None
        else:
            src_ip = packets[0][IP].src
            dst_ip = packets[0][IP].dst
            print(f"Flow direction set: Source IP = {src_ip}, Destination IP = {dst_ip}")
    except (IndexError, KeyError, AttributeError) as e:
        print(f"Error determining flow direction: {e}")
        return None  # No valid IP packets or other issue
    
    last_fwd_time = None
    last_bwd_time = None
    
    valid_packet_count = 0
    for pkt in packets:
        if IP not in pkt:
            print(f"Warning: Packet {valid_packet_count} has no IP layer. Skipping.")
            continue
        valid_packet_count += 1
        packet_lengths.append(len(pkt))
        
        # Forward or backward packet
        if src_ip and dst_ip:
            if pkt[IP].src == src_ip:
                fwd_packets += 1
                fwd_packet_lengths.append(len(pkt))
                if last_fwd_time:
                    fwd_iat.append(pkt.time - last_fwd_time)
                last_fwd_time = pkt.time
            elif pkt[IP].dst == src_ip:  # Backward packet
                bwd_packets += 1
                bwd_packet_lengths.append(len(pkt))
                if last_bwd_time:
                    bwd_iat.append(pkt.time - last_bwd_time)
                last_bwd_time = pkt.time
    
    print(f"Processed {valid_packet_count} valid packets: Fwd = {fwd_packets}, Bwd = {bwd_packets}")
    if not packet_lengths:
        print("Error: No valid packet lengths extracted.")
        return None
    
    # Calculate features
    features = {
        'Flow Duration': flow_duration,
        'Total Fwd Packets': fwd_packets,
        'Total Backward Packets': bwd_packets,
        'Fwd Packet Length Mean': np.mean(fwd_packet_lengths) if fwd_packet_lengths else 0,
        'Bwd Packet Length Mean': np.mean(bwd_packet_lengths) if bwd_packet_lengths else 0,
        'Flow Bytes/s': (sum(fwd_packet_lengths) + sum(bwd_packet_lengths)) / (duration if duration > 0 else 1),
        'Flow Packets/s': (valid_packet_count) / (duration if duration > 0 else 1),
        'Fwd IAT Mean': np.mean(fwd_iat) if fwd_iat else 0,
        'Bwd IAT Mean': np.mean(bwd_iat) if bwd_iat else 0,
        'Packet Length Mean': np.mean(packet_lengths) if packet_lengths else 0
    }
    
    return pd.DataFrame([features])

if __name__ == "__main__":
    # Test feature extraction
    packets = sniff(count=10, timeout=5)  # Capture 10 packets
    features = extract_features(packets)
    print(features)