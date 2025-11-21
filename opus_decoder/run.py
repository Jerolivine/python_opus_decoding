import opuslib
import numpy as np
import struct
import wave


def decode_opus_from_file(file_path, sample_rate=16000, channels=2, frame_duration_ms=20):
    """
    Decode Opus data from a text file containing integers (comma-separated)
    """
    # Read integers from file
    with open(file_path, 'r') as f:
        data_string = f.read()
        string_list = data_string.strip().split(',')
        opus_integers = [int(s) for s in string_list if s.strip()]

    print(f"Read {len(opus_integers)} opus bytes")

    # Convert integers to bytes
    opus_bytes = bytes(opus_integers)

    return decode_opus_with_lengths(opus_bytes, sample_rate, channels, frame_duration_ms)


def decode_opus_with_lengths(opus_data: bytes, sample_rate: int, channels: int, frame_duration_ms: int) -> np.ndarray:
    decoder = opuslib.Decoder(sample_rate, channels)

    # Frame size is samples PER CHANNEL
    frame_size = (sample_rate // 1000) * frame_duration_ms  # 16000/1000 * 20 = 960

    print(f"Frame size: {frame_size} samples per channel")

    all_pcm = []
    offset = 0
    packet_count = 0

    while offset < len(opus_data):
        # Read packet length (4 bytes, little-endian u32)
        if offset + 4 > len(opus_data):
            print(f"Warning: Incomplete packet length at offset {offset}")
            break

        packet_len = struct.unpack('<I', opus_data[offset:offset + 4])[0]
        print(f"Packet {packet_count}: length={packet_len}")
        offset += 4

        if offset + packet_len > len(opus_data):
            print(f"Warning: Incomplete packet data at offset {offset}, expected {packet_len} bytes")
            break

        # Read packet data
        packet = opus_data[offset:offset + packet_len]
        offset += packet_len

        # Decode packet
        try:
            # decode_float returns bytes, need to convert to float32
            pcm_bytes = decoder.decode_float(packet, frame_size)

            # Convert bytes to float32 array
            # Each float32 is 4 bytes
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.float32)

            print(f"Decoded {len(pcm_array)} samples from packet {packet_count}")
            all_pcm.append(pcm_array)
            packet_count += 1
        except Exception as e:
            print(f"Error decoding packet {packet_count}: {e}")
            import traceback
            traceback.print_exc()
            break

    print(f"Decoded {packet_count} packets")

    # Concatenate all PCM data
    if all_pcm:
        result = np.concatenate(all_pcm)
        print(f"Total samples: {len(result)}")
        print(f"Expected duration: {len(result) / (sample_rate * channels):.2f} seconds")
        return result
    else:
        return np.array([], dtype=np.float32)


def save_as_wav(pcm_data, output_path, sample_rate=16000, channels=2):
    """Save decoded PCM data as WAV file"""

    # Convert float32 (-1.0 to 1.0) to int16
    pcm_int16 = (pcm_data * 32767.0).clip(-32768, 32767).astype(np.int16)

    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_int16.tobytes())

    duration = len(pcm_data) / (sample_rate * channels)
    print(f"Saved WAV file: {output_path}")
    print(f"Duration: {duration:.2f} seconds")


# Usage
try:
    pcm_audio = decode_opus_from_file(
        'opus_data 16000-mic.txt',
        sample_rate=16000,
        channels=2,
        frame_duration_ms=20
    )

    print(f"\nFinal PCM array length: {len(pcm_audio)}")

    # Save as WAV
    save_as_wav(pcm_audio, 'decoded_output.wav', sample_rate=16000, channels=2)

except Exception as e:
    print(f"Decoding failed: {e}")
    import traceback

    traceback.print_exc()