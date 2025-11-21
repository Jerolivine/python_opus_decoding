use core::panic;
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    sync::{Arc, Mutex},
};

use audiopus::{Application, Channels, SampleRate, coder::Encoder};
use cpal::{
    FromSample, Sample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

const TARGET_SAMPLE_RATE: u32 = 48000;
const FRAME_DURATION_MS: usize = 20;

struct OpusFileWriter {
    file: File,
    serial: u32,
    page_sequence: u32,
    granule_position: u64,
    frame_size: usize,
}

impl OpusFileWriter {
    fn new(path: &str, sample_rate: u32, channels: u8) -> std::io::Result<Self> {
        let mut writer = OpusFileWriter {
            file: File::create(path)?,
            serial: rand::random(),
            page_sequence: 0,
            granule_position: 0,
            frame_size: match sample_rate {
                48000 => 960,
                24000 => 480,
                16000 => 320,
                _ => 960,
            },
        };

        // Write Opus identification header
        writer.write_id_header(sample_rate, channels)?;

        // Write Opus comment header
        writer.write_comment_header()?;

        Ok(writer)
    }

    fn write_id_header(&mut self, sample_rate: u32, channels: u8) -> std::io::Result<()> {
        let mut header = Vec::new();
        header.extend_from_slice(b"OpusHead");
        header.push(1); // Version
        header.push(channels);
        header.extend_from_slice(&3840u16.to_le_bytes()); // Pre-skip (80ms at 48kHz)
        header.extend_from_slice(&sample_rate.to_le_bytes());
        header.extend_from_slice(&0u16.to_le_bytes()); // Output gain
        header.push(0); // Channel mapping family

        self.write_ogg_page(&header, 0, 0x02)?; // BOS flag
        Ok(())
    }

    fn write_comment_header(&mut self) -> std::io::Result<()> {
        let vendor = b"Rust Opus Encoder";
        let mut header = Vec::new();
        header.extend_from_slice(b"OpusTags");
        header.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        header.extend_from_slice(vendor);
        header.extend_from_slice(&0u32.to_le_bytes()); // No user comments

        self.write_ogg_page(&header, 0, 0)?;
        Ok(())
    }

    fn write_audio_packet(&mut self, packet: &[u8]) -> std::io::Result<()> {
        self.granule_position += self.frame_size as u64;
        self.write_ogg_page(packet, self.granule_position, 0)?;
        Ok(())
    }

    fn finalize(&mut self) -> std::io::Result<()> {
        // Write empty page with EOS flag
        self.write_ogg_page(&[], self.granule_position, 0x04)?;
        Ok(())
    }

    fn write_ogg_page(&mut self, data: &[u8], granule: u64, flags: u8) -> std::io::Result<()> {
        let num_segments = ((data.len() + 254) / 255).max(1);
        let mut segments = vec![255u8; num_segments];

        if !data.is_empty() {
            segments[num_segments - 1] = (data.len() % 255) as u8;
            if segments[num_segments - 1] == 0 {
                segments[num_segments - 1] = 255;
            }
        } else {
            segments[0] = 0;
        }

        let mut page = Vec::new();
        page.extend_from_slice(b"OggS"); // Capture pattern
        page.push(0); // Version
        page.push(flags); // Header type
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&self.serial.to_le_bytes());
        page.extend_from_slice(&self.page_sequence.to_le_bytes());
        page.extend_from_slice(&[0u8; 4]); // CRC (placeholder)
        page.push(segments.len() as u8);
        page.extend_from_slice(&segments);
        page.extend_from_slice(data);

        // Calculate CRC32
        let crc = Self::crc32(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());

        self.file.write_all(&page)?;
        self.page_sequence += 1;
        Ok(())
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0u32;
        for &byte in data {
            crc ^= (byte as u32) << 24;
            for _ in 0..8 {
                if crc & 0x80000000 != 0 {
                    crc = (crc << 1) ^ 0x04c11db7;
                } else {
                    crc <<= 1;
                }
            }
        }
        crc
    }
}

// fn encode_to_opus_file(
//     pcm_data: Vec<f32>,
//     sample_rate: u32,
//     channels: Channels,
//     output_path: &str,
// ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
//     let mut encoder = Encoder::new(sample_rate, channels, Application::Audio)?;

//     let pcm_i16: Vec<i16> = pcm_data
//         .iter()
//         .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
//         .collect();

//     let frame_size: usize = 960;
//     let ch_count = if matches!(channels, Channels::Stereo) {
//         2
//     } else {
//         1
//     };

//     let samples_per_frame = frame_size * ch_count;
//     let mut encoded_frame = vec![0u8; 4000];

//     let mut file = std::fs::File::create(output_path)?;
//     let mut all_bytes = Vec::new();

//     for chunk in pcm_i16.chunks(samples_per_frame) {
//         if chunk.len() == samples_per_frame {
//             let len = encoder.encode(chunk, &mut encoded_frame)?;
//             all_bytes.extend_from_slice(&encoded_frame[..len]);
//         }
//     }

//     use std::io::Write;
//     writeln!(
//         file,
//         "{}",
//         all_bytes
//             .iter()
//             .map(|l| l.to_string())
//             .collect::<Vec<_>>()
//             .join(",")
//     )?;

//     Ok(all_bytes)
// }

// pub fn encode_to_opus(pcm: Vec<f32>) -> anyhow::Result<Vec<u8>> {
//     let sample_rate = get_sample_rate()?;
//     let frame_size = get_frame_size(sample_rate);
//     let encoder = Encoder::new(sample_rate, Channels::Stereo, Application::Audio)?;

//     println!("frame size: {}", frame_size);

//     let mut offset = 0;
//     let mut encoded = vec![0u8; 4000];
//     let mut opus_data: Vec<u8> = Vec::new();

//     while offset + frame_size <= pcm.len() {
//         let frame = &pcm[offset..offset + frame_size];
//         let packet_len = encoder.encode_float(frame, &mut encoded)?;
       

//         let packet = &encoded[..packet_len];
//         opus_data.extend_from_slice(packet);
//         offset += frame_size;
//     }

//     let mut file = std::fs::File::create("opus_data.txt")?;
//     use std::io::Write;
//     writeln!(
//         file,
//         "{}",
//         opus_data
//             .iter()
//             .map(|l| l.to_string())
//             .collect::<Vec<_>>()
//             .join(",")
//     )?;

//     Ok(opus_data)
// }

pub fn encode_to_opus_new(pcm: Vec<f32>) -> anyhow::Result<Vec<u8>> {
    let sample_rate = get_sample_rate()?;
    let frame_size = get_frame_size(sample_rate);
    let encoder = Encoder::new(sample_rate, Channels::Stereo, Application::Audio)?;

    println!("frame size: {}", frame_size);

    let mut offset = 0;
    let mut encoded = vec![0u8; 4000];
    let mut opus_data: Vec<u8> = Vec::new();

    while offset + frame_size <= pcm.len() {
        let frame = &pcm[offset..offset + frame_size];
        let packet_len = encoder.encode_float(frame, &mut encoded)?;
        println!("packet_len {}", packet_len);

        let packet = &encoded[..packet_len];

        // Store length as 4-byte prefix (u32)
        let e_length = &(packet_len as u32).to_le_bytes();

        opus_data.extend_from_slice(e_length);
        opus_data.extend_from_slice(packet);

        offset += frame_size;
    }

    let mut file = std::fs::File::create("opus_data.txt")?;
    use std::io::Write;
    writeln!(
        file,
        "{}",
        opus_data
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )?;

    Ok(opus_data)
}

fn get_sample_rate() -> anyhow::Result<SampleRate> {
    match TARGET_SAMPLE_RATE {
        8000 => Ok(SampleRate::Hz8000),
        12000 => Ok(SampleRate::Hz12000),
        16000 => Ok(SampleRate::Hz16000),
        24000 => Ok(SampleRate::Hz24000),
        48000 => Ok(SampleRate::Hz48000),
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported sample rate to encode opus: {}",
                TARGET_SAMPLE_RATE
            ));
        }
    }
}

fn get_frame_size(sample_rate: SampleRate) -> usize {
    match sample_rate {
        SampleRate::Hz8000 => 8 * FRAME_DURATION_MS,
        SampleRate::Hz12000 => 12 * FRAME_DURATION_MS,
        SampleRate::Hz16000 => 160 * FRAME_DURATION_MS,
        SampleRate::Hz24000 => 24 * FRAME_DURATION_MS,
        SampleRate::Hz48000 => 48 * FRAME_DURATION_MS,
    }
}

fn main() -> anyhow::Result<()> {
    let audio_data = Arc::new(Mutex::new(Vec::<f32>::new()));
    let audio_data_c = audio_data.clone();

    let host = cpal::default_host();

    // Get the default *output* device (speaker)
    let device = host
        .default_output_device()
        .expect("no output device available");

    // On Windows, output devices can be used in loopback mode to capture their playback.
    let config = device.default_output_config()?.config();

    println!("{:?}", config);

    const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/recorded.wav");
    println!("PATH {}", PATH);
    let spec = wav_spec_from_config(&device.default_output_config().unwrap());
    let writer = hound::WavWriter::create(PATH, spec)?;
    let writer = Arc::new(Mutex::new(Some(writer)));
    let writer_2 = writer.clone();

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            // Encode float samples to Opus

            write_input_data::<f32, f32>(data, &writer_2);

            audio_data.lock().unwrap().extend_from_slice(data);
        },
        |err| eprintln!("Stream error: {err}"),
        None,
    )?;

    stream.play()?;
    std::thread::sleep(std::time::Duration::from_secs(10));

    match encode_to_opus_new(audio_data_c.lock().unwrap().to_vec()) {
        Ok(opus_content) => {
            // println!("{:?}", opus_content.len());
            // Write frame lengths on first line (comma-separated)
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    Ok(())
}

fn wav_spec_from_config(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate().0 as _,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: sample_format(config.sample_format()),
    }
}

fn sample_format(format: cpal::SampleFormat) -> hound::SampleFormat {
    if format.is_float() {
        hound::SampleFormat::Float
    } else {
        hound::SampleFormat::Int
    }
}

type WavWriterHandle = Arc<Mutex<Option<hound::WavWriter<BufWriter<File>>>>>;

fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
where
    T: Sample,
    U: Sample + hound::Sample + FromSample<T>,
{
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input.iter() {
                let sample: U = U::from_sample(sample);
                writer.write_sample(sample).ok();
            }
        }
    }
}
